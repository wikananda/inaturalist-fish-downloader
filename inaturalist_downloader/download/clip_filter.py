"""Optional CLIP-based context filtering for accepted fish images."""

import argparse
import json
import threading
from pathlib import Path
from typing import Optional

from .image_quality import Image, ImageOps, pillow_available
from .model_imports import model_import_context

CLIP_LOCK = threading.Lock()
CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_MODEL_NAME = None
CLIP_CACHE_DIR = None
CLIP_BACKEND = None

DEFAULT_POSITIVE_PROMPTS = [
    "an underwater photo of a fish",
    "a fish swimming underwater",
    "a fish in a natural aquatic environment",
    "a fish swimming in a river",
    "a fish in murky water",
]

DEFAULT_NEGATIVE_PROMPTS = [
    "two or more fishes swimming closely together",
    "a person holding a fish",
    "a fish out of water",
    "a fish on a fishing rod",
    "a fish caught by a hook",
    "a dead fish on a table",
    "a fish in a market",
    "a cooked fish on a plate",
    "a person fishing",
    "a hand holding a fish",
]


def _transformers_error_message(exc: Exception) -> str:
    """Build an actionable Transformers import error message."""
    import sys

    return (
        "CLIP filtering requires a working Transformers install in the current "
        f"Python interpreter ({sys.executable}). Original import error: "
        f"{type(exc).__name__}: {exc}"
    )


def validate_clip_import(backend: str = "clip") -> None:
    """Fail early if semantic-filter dependencies cannot import."""
    try:
        import torch  # noqa: F401
        with model_import_context():
            if backend == "siglip2":
                from transformers import AutoModel, AutoProcessor  # noqa: F401
            else:
                from transformers import CLIPModel, CLIPProcessor  # noqa: F401
    except Exception as exc:
        raise RuntimeError(_transformers_error_message(exc)) from exc


def get_clip_components(
    model_name: str,
    cache_dir: Optional[str],
    backend: str = "clip",
):
    """Load and cache CLIP/SigLIP model and processor instances."""
    global CLIP_MODEL, CLIP_PROCESSOR, CLIP_MODEL_NAME, CLIP_CACHE_DIR, CLIP_BACKEND

    with CLIP_LOCK:
        if (
            CLIP_MODEL is not None
            and CLIP_PROCESSOR is not None
            and CLIP_MODEL_NAME == model_name
            and CLIP_CACHE_DIR == cache_dir
            and CLIP_BACKEND == backend
        ):
            return CLIP_MODEL, CLIP_PROCESSOR

        try:
            with model_import_context():
                if backend == "siglip2":
                    from transformers import AutoModel, AutoProcessor
                else:
                    from transformers import CLIPModel, CLIPProcessor

                load_kwargs = {}
                if cache_dir:
                    cache_path = Path(cache_dir)
                    cache_path.mkdir(parents=True, exist_ok=True)
                    load_kwargs["cache_dir"] = str(cache_path)

                def load_components(source, kwargs):
                    if backend == "siglip2":
                        processor = AutoProcessor.from_pretrained(source, **kwargs)
                        model = AutoModel.from_pretrained(source, **kwargs)
                    else:
                        processor = CLIPProcessor.from_pretrained(source, **kwargs)
                        model = CLIPModel.from_pretrained(source, **kwargs)
                    return model, processor

                try:
                    model, processor = load_components(model_name, load_kwargs)
                except Exception as online_exc:
                    # Transformers/HF Hub can fail a metadata HEAD request even
                    # when every model file is already cached. Retry without any
                    # network access before reporting the original load error.
                    try:
                        snapshot = _cached_snapshot_path(model_name, cache_dir)
                        local_source = str(snapshot) if snapshot else model_name
                        model, processor = load_components(
                            local_source,
                            {**load_kwargs, "local_files_only": True},
                        )
                    except Exception:
                        raise online_exc
        except Exception as exc:
            raise RuntimeError(_transformers_error_message(exc)) from exc

        CLIP_MODEL = model
        CLIP_PROCESSOR = processor
        CLIP_MODEL_NAME = model_name
        CLIP_CACHE_DIR = cache_dir
        CLIP_BACKEND = backend
        return CLIP_MODEL, CLIP_PROCESSOR


def _processor_kwargs(backend: str) -> dict:
    if backend == "siglip2":
        return {"padding": "max_length", "max_length": 64}
    return {"padding": True}


def _cached_snapshot_path(model_name: str, cache_dir: Optional[str]) -> Optional[Path]:
    """Resolve a Hugging Face model ID to a complete local snapshot if cached."""
    if not cache_dir or "/" not in model_name:
        return None
    model_cache = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    revision = None
    main_ref = model_cache / "refs" / "main"
    if main_ref.exists():
        revision = main_ref.read_text(encoding="utf-8").strip()
    if revision:
        snapshot = model_cache / "snapshots" / revision
        if snapshot.is_dir():
            return snapshot
    snapshots = model_cache / "snapshots"
    if snapshots.is_dir():
        for snapshot in sorted(snapshots.iterdir(), reverse=True):
            if snapshot.is_dir() and (snapshot / "config.json").exists():
                return snapshot
    return None


def load_clip_prompts(path: Optional[str]) -> tuple[list[str], list[str]]:
    """Load CLIP prompts from JSON or fall back to bundled defaults.

    The JSON file must contain:

    {
      "positive": ["..."],
      "negative": ["..."]
    }
    """
    if not path:
        return list(DEFAULT_POSITIVE_PROMPTS), list(DEFAULT_NEGATIVE_PROMPTS)

    prompts_path = Path(path)
    if not prompts_path.exists():
        raise FileNotFoundError(f"CLIP prompts file not found: {prompts_path}")

    payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    positive = [str(item).strip() for item in payload.get("positive", []) if str(item).strip()]
    negative = [str(item).strip() for item in payload.get("negative", []) if str(item).strip()]

    if not positive:
        raise ValueError("CLIP prompts file must contain at least one positive prompt")
    if not negative:
        raise ValueError("CLIP prompts file must contain at least one negative prompt")
    return positive, negative


def resolve_clip_device(args: argparse.Namespace):
    """Resolve CLIP device from CLI arguments or select CPU."""
    import torch

    if args.clip_device:
        return torch.device(args.clip_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preload_clip_model(args: argparse.Namespace) -> str:
    """Download + load the CLIP model/processor and place it on the target device.

    Ensures the CLIP weights are fetched and the model is built and warmed before any
    image download begins, instead of lazily on the first crop inside a worker thread.
    """
    try:
        backend = getattr(args, "clip_backend", "clip")
        if backend == "clip":
            validate_clip_import()
        else:
            validate_clip_import(backend)
    except RuntimeError:
        raise
    try:
        import torch

        model, processor = get_clip_components(
            args.clip_model, args.clip_cache_dir, backend
        )
        device = resolve_clip_device(args)
        model = model.to(device)
        model.eval()
        # Warmup: a tiny text+image forward so tokenizer/vision weights are fully
        # initialized and any remaining lazy download happens up front.
        if pillow_available():
            warmup_image = Image.new("RGB", (64, 64), color=(127, 127, 127))
            inputs = processor(
                text=["a fish"],
                images=warmup_image,
                return_tensors="pt",
                **_processor_kwargs(backend),
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with CLIP_LOCK:
                with torch.no_grad():
                    model(**inputs)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"CLIP model failed to load/warm up: {type(exc).__name__}: {exc}"
        ) from exc
    return args.clip_model


def run_clip_filter(
    image_path: Path,
    args: argparse.Namespace,
) -> tuple[bool, Optional[str], dict]:
    """Run CLIP prompt scoring and accept/reject by score margin."""
    return run_clip_filter_batch([image_path], args)[0]


def run_clip_filter_batch(
    image_paths: list[Path],
    args: argparse.Namespace,
) -> list[tuple[bool, Optional[str], dict]]:
    """Score a batch of images with one CLIP forward pass."""
    if not image_paths:
        return []
    if not pillow_available():
        return [
            (False, "pillow_not_installed", {"enabled": True})
            for _ in image_paths
        ]

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(_transformers_error_message(exc)) from exc

    positive_prompts = args.clip_positive_prompts
    negative_prompts = args.clip_negative_prompts
    all_prompts = positive_prompts + negative_prompts

    backend = getattr(args, "clip_backend", "clip")
    model, processor = get_clip_components(
        args.clip_model, args.clip_cache_dir, backend
    )
    device = resolve_clip_device(args)
    model = model.to(device)
    model.eval()

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            images.append(image.copy())

    try:
        inputs = processor(
            text=all_prompts,
            images=images,
            return_tensors="pt",
            **_processor_kwargs(backend),
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with CLIP_LOCK:
            with torch.no_grad():
                outputs = model(**inputs)
    finally:
        for image in images:
            image.close()

    logits_tensor = outputs.logits_per_image.detach().cpu()
    if backend == "siglip2":
        all_scores = torch.sigmoid(logits_tensor).tolist()
        score_kind = "sigmoid_probability_margin"
    else:
        all_scores = logits_tensor.tolist()
        score_kind = "logit_margin"
    results = []
    for scores in all_scores:
        prompt_scores = {
            prompt: round(float(score), 6)
            for prompt, score in zip(all_prompts, scores)
        }
        positive_scores = scores[: len(positive_prompts)]
        negative_scores = scores[len(positive_prompts) :]
        positive_max = max(float(score) for score in positive_scores)
        negative_max = max(float(score) for score in negative_scores)
        context_score = positive_max - negative_max
        metrics = {
            "enabled": True,
            "backend": backend,
            "model": args.clip_model,
            "cache_dir": args.clip_cache_dir,
            "device": str(device),
            "threshold": args.clip_threshold,
            "score_kind": score_kind,
            "positive_prompt_count": len(positive_prompts),
            "negative_prompt_count": len(negative_prompts),
            "positive_max_score": round(positive_max, 6),
            "negative_max_score": round(negative_max, 6),
            "context_score": round(context_score, 6),
            "prompt_scores": prompt_scores,
        }
        if context_score < args.clip_threshold:
            results.append((False, "clip_filtered", metrics))
        else:
            results.append((True, None, metrics))
    return results
