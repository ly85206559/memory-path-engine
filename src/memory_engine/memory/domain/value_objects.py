from __future__ import annotations

from dataclasses import dataclass


def _bounded(value: float) -> float:
    return max(0.0, min(value, 1.0))


@dataclass(frozen=True, slots=True)
class PalaceLocation:
    building: str
    floor: str | None = None
    room: str | None = None
    locus: str | None = None

    def as_key(self) -> str:
        parts = [self.building, self.floor, self.room, self.locus]
        return "/".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class SalienceProfile:
    importance: float
    risk: float
    novelty: float
    confidence: float
    emotional_intensity: float = 0.0
    recency: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "importance",
            "risk",
            "novelty",
            "confidence",
            "emotional_intensity",
            "recency",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")

    def base_score(self) -> float:
        score = (
            self.importance * 0.28
            + self.risk * 0.22
            + self.novelty * 0.16
            + self.confidence * 0.16
            + self.emotional_intensity * 0.08
            + self.recency * 0.10
        )
        return _bounded(score)
