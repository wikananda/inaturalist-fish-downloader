"""Optional CLIP-based context filtering for accepted fish images."""

import argparse
import json
import threading
from pathlib import Path
from typing import Optional

from .image_quality import Image, ImageOps, pillow_available

CLIP_LOCK = threading.Lock()
CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_MODEL_NAME = None
CLIP_CACHE_DIR = None

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


def validate_clip_import() -> None:
    """Fail early if CLIP dependencies cannot import in the active interpreter."""
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor  # noqa: F401
    except Exception as exc:
        raise RuntimeError(_transformers_error_message(exc)) from exc


def get_clip_components(model_name: str, cache_dir: Optional[str]):
    """Load and cache CLIP model and processor instances."""
    global CLIP_MODEL, CLIP_PROCESSOR, CLIP_MODEL_NAME, CLIP_CACHE_DIR

    with CLIP_LOCK:
        if (
            CLIP_MODEL is not None
            and CLIP_PROCESSOR is not None
            and CLIP_MODEL_NAME == model_name
            and CLIP_CACHE_DIR == cache_dir
        ):
            return CLIP_MODEL, CLIP_PROCESSOR

        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:
            raise RuntimeError(_transformers_error_message(exc)) from exc

        load_kwargs = {}
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            load_kwargs["cache_dir"] = str(cache_path)

        processor = CLIPProcessor.from_pretrained(model_name, **load_kwargs)
        model = CLIPModel.from_pretrained(model_name, **load_kwargs)

        CLIP_MODEL = model
        CLIP_PROCESSOR = processor
        CLIP_MODEL_NAME = model_name
        CLIP_CACHE_DIR = cache_dir
        return CLIP_MODEL, CLIP_PROCESSOR


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


def run_clip_filter(
    image_path: Path,
    args: argparse.Namespace,
) -> tuple[bool, Optional[str], dict]:
    """Run CLIP prompt scoring and accept/reject by score margin."""
    if not pillow_available():
        return False, "pillow_not_installed", {"enabled": True}

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(_transformers_error_message(exc)) from exc

    positive_prompts = args.clip_positive_prompts
    negative_prompts = args.clip_negative_prompts
    all_prompts = positive_prompts + negative_prompts

    model, processor = get_clip_components(args.clip_model, args.clip_cache_dir)
    device = resolve_clip_device(args)
    model = model.to(device)
    model.eval()

    with Image.open(image_path) as source_image:
        image = ImageOps.exif_transpose(source_image)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")

        inputs = processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with CLIP_LOCK:
            with torch.no_grad():
                outputs = model(**inputs)

    logits = outputs.logits_per_image[0].detach().cpu().tolist()
    prompt_scores = {
        prompt: round(float(score), 6) for prompt, score in zip(all_prompts, logits)
    }

    positive_scores = logits[: len(positive_prompts)]
    negative_scores = logits[len(positive_prompts) :]
    positive_max = max(float(score) for score in positive_scores)
    negative_max = max(float(score) for score in negative_scores)
    context_score = positive_max - negative_max

    metrics = {
        "enabled": True,
        "model": args.clip_model,
        "cache_dir": args.clip_cache_dir,
        "device": str(device),
        "threshold": args.clip_threshold,
        "positive_prompt_count": len(positive_prompts),
        "negative_prompt_count": len(negative_prompts),
        "positive_max_score": round(positive_max, 6),
        "negative_max_score": round(negative_max, 6),
        "context_score": round(context_score, 6),
        "prompt_scores": prompt_scores,
    }

    if context_score < args.clip_threshold:
        return False, "clip_filtered", metrics

    return True, None, metrics
