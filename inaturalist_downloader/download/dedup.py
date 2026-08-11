"""Dataset-wide accepted-image identity and duplicate protection."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the exact content digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path, hash_size: int = 8) -> str:
    """Return a 64-bit perceptual difference hash without extra dependencies."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    value = 0
    width = hash_size + 1
    for row in range(hash_size):
        offset = row * width
        for column in range(hash_size):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    """Return bit distance between two hexadecimal hashes."""
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _hash_segments(value: str, parts: int, bit_count: int = 64) -> list[int]:
    """Split a hash into bands for exact-band Hamming-radius lookup."""
    numeric = int(value, 16)
    base_width, remainder = divmod(bit_count, parts)
    segments = []
    shift = 0
    for index in range(parts):
        width = base_width + int(index < remainder)
        segments.append((numeric >> shift) & ((1 << width) - 1))
        shift += width
    return segments


@dataclass(frozen=True)
class DuplicateDecision:
    """Result of attempting to reserve an accepted image globally."""

    accepted: bool
    reason: str | None
    metrics: dict[str, Any]


class DatasetDeduplicator:
    """Thread-safe identity registry shared by every species worker.

    The accepted manifest is loaded on startup so resume runs remain protected.
    Exact and perceptual hashes are computed for legacy rows that predate stored
    hashes whenever their accepted output is still available.
    """

    def __init__(
        self,
        accepted_manifest: Path,
        *,
        enabled: bool = True,
        observation_ids: bool = True,
        photo_ids: bool = True,
        exact_content: bool = True,
        perceptual_content: bool = True,
        perceptual_distance: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.check_observation_ids = bool(observation_ids)
        self.check_photo_ids = bool(photo_ids)
        self.check_exact_content = bool(exact_content)
        self.check_perceptual_content = bool(perceptual_content)
        self.perceptual_distance = int(perceptual_distance)
        self._lock = threading.Lock()
        self._observations: dict[int, dict[str, Any]] = {}
        self._photos: dict[int, dict[str, Any]] = {}
        self._exact_hashes: dict[str, dict[str, Any]] = {}
        self._perceptual_hashes: dict[str, dict[str, Any]] = {}
        self._perceptual_bands: dict[tuple[int, int], set[str]] = {}
        if self.enabled:
            self._load_manifest(Path(accepted_manifest))

    @staticmethod
    def _species(record: dict[str, Any]) -> str:
        return str(
            record.get("canonical_name") or record.get("species_name") or "unknown"
        )

    @staticmethod
    def _identity(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "species": DatasetDeduplicator._species(record),
            "taxon_id": record.get("taxon_id"),
            "observation_id": record.get("observation_id"),
            "photo_id": record.get("photo_id"),
            "saved_output_path": record.get("saved_output_path")
            or record.get("target_output_path"),
        }

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if not value:
                    continue
                try:
                    record = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record

    @staticmethod
    def _existing_output(record: dict[str, Any]) -> Path | None:
        for field in ("saved_output_path", "target_output_path"):
            value = record.get(field)
            if value and Path(value).is_file():
                return Path(value)
        return None

    def _load_manifest(self, path: Path) -> None:
        for record in self._iter_jsonl(path):
            identity = self._identity(record)
            observation_id = record.get("observation_id")
            photo_id = record.get("photo_id")
            if observation_id is not None:
                self._observations.setdefault(int(observation_id), identity)
            if photo_id is not None:
                self._photos.setdefault(int(photo_id), identity)

            exact_hash = record.get("content_sha256")
            perceptual_hash = record.get("perceptual_dhash")
            if (
                (self.check_exact_content and not exact_hash)
                or (self.check_perceptual_content and not perceptual_hash)
            ):
                output_path = self._existing_output(record)
                if output_path is not None:
                    try:
                        exact_hash = exact_hash or sha256_file(output_path)
                        perceptual_hash = perceptual_hash or difference_hash(output_path)
                    except (OSError, ValueError):
                        pass
            if exact_hash:
                self._exact_hashes.setdefault(str(exact_hash), identity)
            if perceptual_hash:
                self._register_perceptual_hash(str(perceptual_hash), identity)

    def _register_perceptual_hash(
        self, value: str, identity: dict[str, Any]
    ) -> None:
        if value in self._perceptual_hashes:
            return
        self._perceptual_hashes[value] = identity
        if self.perceptual_distance >= 64:
            return
        part_count = self.perceptual_distance + 1
        for index, segment in enumerate(_hash_segments(value, part_count)):
            self._perceptual_bands.setdefault((index, segment), set()).add(value)

    def _near_hash_candidates(self, value: str) -> set[str]:
        if self.perceptual_distance >= 64:
            return set(self._perceptual_hashes)
        candidates: set[str] = set()
        part_count = self.perceptual_distance + 1
        for index, segment in enumerate(_hash_segments(value, part_count)):
            candidates.update(self._perceptual_bands.get((index, segment), set()))
        return candidates

    def _decision(
        self,
        *,
        reason_base: str,
        record: dict[str, Any],
        existing: dict[str, Any],
        metrics: dict[str, Any],
    ) -> DuplicateDecision:
        cross_species = self._species(record).casefold() != str(
            existing.get("species") or ""
        ).casefold()
        prefix = "conflicting" if cross_species else "duplicate"
        return DuplicateDecision(
            accepted=False,
            reason=f"{prefix}_{reason_base}",
            metrics={
                **metrics,
                "cross_species": cross_species,
                "matched_record": existing,
            },
        )

    def check_source_identity(self, record: dict[str, Any]) -> DuplicateDecision:
        """Reject already-accepted observation/photo IDs before downloading."""
        if not self.enabled:
            return DuplicateDecision(True, None, {"enabled": False})
        with self._lock:
            observation_id = record.get("observation_id")
            if self.check_observation_ids and observation_id is not None:
                existing = self._observations.get(int(observation_id))
                if existing is not None:
                    return self._decision(
                        reason_base="observation_id",
                        record=record,
                        existing=existing,
                        metrics={"enabled": True},
                    )
            photo_id = record.get("photo_id")
            if self.check_photo_ids and photo_id is not None:
                existing = self._photos.get(int(photo_id))
                if existing is not None:
                    return self._decision(
                        reason_base="photo_id",
                        record=record,
                        existing=existing,
                        metrics={"enabled": True},
                    )
        return DuplicateDecision(True, None, {"enabled": True})

    def check_and_register(
        self,
        record: dict[str, Any],
        image_path: Path,
    ) -> DuplicateDecision:
        """Atomically check all identities and reserve a newly accepted image."""
        if not self.enabled:
            return DuplicateDecision(True, None, {"enabled": False})

        exact_hash = sha256_file(image_path) if self.check_exact_content else None
        perceptual_hash = (
            difference_hash(image_path) if self.check_perceptual_content else None
        )
        metrics: dict[str, Any] = {
            "enabled": True,
            "content_sha256": exact_hash,
            "perceptual_dhash": perceptual_hash,
            "perceptual_distance_threshold": self.perceptual_distance,
        }
        identity = self._identity(record)

        with self._lock:
            observation_id = record.get("observation_id")
            if self.check_observation_ids and observation_id is not None:
                existing = self._observations.get(int(observation_id))
                if existing is not None:
                    return self._decision(
                        reason_base="observation_id",
                        record=record,
                        existing=existing,
                        metrics=metrics,
                    )
            photo_id = record.get("photo_id")
            if self.check_photo_ids and photo_id is not None:
                existing = self._photos.get(int(photo_id))
                if existing is not None:
                    return self._decision(
                        reason_base="photo_id",
                        record=record,
                        existing=existing,
                        metrics=metrics,
                    )
            if exact_hash is not None:
                existing = self._exact_hashes.get(exact_hash)
                if existing is not None:
                    return self._decision(
                        reason_base="exact_content",
                        record=record,
                        existing=existing,
                        metrics=metrics,
                    )
            if perceptual_hash is not None:
                for known_hash in self._near_hash_candidates(perceptual_hash):
                    existing = self._perceptual_hashes[known_hash]
                    distance = hamming_distance(perceptual_hash, known_hash)
                    if distance <= self.perceptual_distance:
                        return self._decision(
                            reason_base="near_content",
                            record=record,
                            existing=existing,
                            metrics={**metrics, "matched_perceptual_distance": distance},
                        )

            if observation_id is not None:
                self._observations[int(observation_id)] = identity
            if photo_id is not None:
                self._photos[int(photo_id)] = identity
            if exact_hash is not None:
                self._exact_hashes[exact_hash] = identity
            if perceptual_hash is not None:
                self._register_perceptual_hash(perceptual_hash, identity)

        return DuplicateDecision(True, None, metrics)
