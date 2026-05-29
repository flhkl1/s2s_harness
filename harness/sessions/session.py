from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Domain:
    name: str
    target_turns: int
    turns_completed: int = 0


@dataclass
class Session:
    persona: str
    domains: list[Domain]
    probe_depth: int
    opening_line: str
    active_domain_idx: int = 0
    _extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Session":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            persona=cfg["persona"].strip(),
            domains=[Domain(**d) for d in cfg["domains"]],
            probe_depth=cfg.get("probe_depth", 2),
            opening_line=cfg.get("opening_line", "").strip(),
        )

    def build_system_prompt(self) -> str:
        domain_list = ", ".join(d.name for d in self.domains)
        return (
            f"{self.persona}\n\n"
            f"Cover these domains across the conversation: {domain_list}. "
            f"Spend roughly {self.probe_depth} follow-up probes per domain before "
            f"transitioning naturally to the next. "
            f"Begin with: \"{self.opening_line}\""
        )

    @property
    def active_domain(self) -> Domain | None:
        if self.active_domain_idx < len(self.domains):
            return self.domains[self.active_domain_idx]
        return None

    def advance_domain(self) -> None:
        self.active_domain_idx = min(
            self.active_domain_idx + 1, len(self.domains) - 1
        )
