"""Persistent per-species progress for resumable accepted-target downloads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STATE_VERSION = 1


def search_scope_key(place_id: int | None, license_code: str | None) -> str:
    place = "global" if place_id is None else str(place_id)
    license_value = license_code or "any"
    return f"place={place}|license={license_value}"


@dataclass
class SpeciesProgress:
    signature: dict[str, Any]
    next_pages: dict[str, int]
    exhausted_scopes: dict[str, bool]
    seen_photo_ids: set[int] = field(default_factory=set)
    seen_observation_ids: set[int] = field(default_factory=set)
    accepted_observation_ids: set[int] = field(default_factory=set)
    accepted_crops_by_observation: dict[str, int] = field(default_factory=dict)
    accepted_outputs: int = 0
    accepted_by_user: dict[str, int] = field(default_factory=dict)
    candidates_scanned: int = 0
    downloaded: int = 0
    download_failed: int = 0
    rejected: int = 0
    unused_valid: int = 0
    batch_index: int = 0
    stop_reason: str | None = None

    def target_count(self, target_unit: str) -> int:
        if target_unit == "observation":
            return len(self.accepted_observation_ids)
        return self.accepted_outputs

    def mark_accepted(self, record: dict[str, Any]) -> None:
        observation_id = record.get("observation_id")
        if observation_id is not None:
            observation_value = int(observation_id)
            self.accepted_observation_ids.add(observation_value)
            observation_key = str(observation_value)
            self.accepted_crops_by_observation[observation_key] = (
                self.accepted_crops_by_observation.get(observation_key, 0) + 1
            )
        self.accepted_outputs += 1
        user_id = record.get("user_id")
        if user_id is not None:
            key = str(user_id)
            self.accepted_by_user[key] = self.accepted_by_user.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "signature": self.signature,
            "next_pages": self.next_pages,
            "exhausted_scopes": self.exhausted_scopes,
            "seen_photo_ids": sorted(self.seen_photo_ids),
            "seen_observation_ids": sorted(self.seen_observation_ids),
            "accepted_observation_ids": sorted(self.accepted_observation_ids),
            "accepted_crops_by_observation": self.accepted_crops_by_observation,
            "accepted_outputs": self.accepted_outputs,
            "accepted_by_user": self.accepted_by_user,
            "candidates_scanned": self.candidates_scanned,
            "downloaded": self.downloaded,
            "download_failed": self.download_failed,
            "rejected": self.rejected,
            "unused_valid": self.unused_valid,
            "batch_index": self.batch_index,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeciesProgress":
        if int(payload.get("state_version", 0)) != STATE_VERSION:
            raise ValueError("Unsupported downloader progress-state version")
        return cls(
            signature=dict(payload["signature"]),
            next_pages={str(key): int(value) for key, value in payload["next_pages"].items()},
            exhausted_scopes={
                str(key): bool(value) for key, value in payload["exhausted_scopes"].items()
            },
            seen_photo_ids={int(value) for value in payload.get("seen_photo_ids", [])},
            seen_observation_ids={
                int(value) for value in payload.get("seen_observation_ids", [])
            },
            accepted_observation_ids={
                int(value) for value in payload.get("accepted_observation_ids", [])
            },
            accepted_crops_by_observation={
                str(key): int(value)
                for key, value in payload.get("accepted_crops_by_observation", {}).items()
            },
            accepted_outputs=int(payload.get("accepted_outputs", 0)),
            accepted_by_user={
                str(key): int(value)
                for key, value in payload.get("accepted_by_user", {}).items()
            },
            candidates_scanned=int(payload.get("candidates_scanned", 0)),
            downloaded=int(payload.get("downloaded", 0)),
            download_failed=int(payload.get("download_failed", 0)),
            rejected=int(payload.get("rejected", 0)),
            unused_valid=int(payload.get("unused_valid", 0)),
            batch_index=int(payload.get("batch_index", 0)),
            stop_reason=payload.get("stop_reason"),
        )


def new_progress(signature: dict[str, Any], scope_keys: list[str]) -> SpeciesProgress:
    return SpeciesProgress(
        signature=signature,
        next_pages={key: 1 for key in scope_keys},
        exhausted_scopes={key: False for key in scope_keys},
    )


def load_progress(
    path: Path,
    *,
    signature: dict[str, Any],
    scope_keys: list[str],
    resume: bool,
) -> SpeciesProgress:
    """Load matching progress, or create a clean state when resume is disabled."""
    if not resume or not path.exists():
        return new_progress(signature, scope_keys)
    payload = json.loads(path.read_text(encoding="utf-8"))
    progress = SpeciesProgress.from_dict(payload)
    if progress.signature != signature:
        raise ValueError(
            f"Resume state does not match the current search configuration: {path}. "
            "Use --no-resume or a new manifest directory."
        )
    for key in scope_keys:
        progress.next_pages.setdefault(key, 1)
        progress.exhausted_scopes.setdefault(key, False)
    return progress


def save_progress(path: Path, progress: SpeciesProgress) -> None:
    """Atomically save progress after a completed candidate batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(progress.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
