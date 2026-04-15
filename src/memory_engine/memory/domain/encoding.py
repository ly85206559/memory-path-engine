from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TriggerProfile:
    phrases: tuple[str, ...] = ()
    situations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EncodingProfile:
    trigger_profile: TriggerProfile = field(default_factory=TriggerProfile)
    scenario_tags: tuple[str, ...] = ()
    symbolic_tags: tuple[str, ...] = ()
