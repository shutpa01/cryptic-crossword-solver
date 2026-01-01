# solver/orchestration/cascade_analyser.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class StageExample:
    clue: str
    answer: str
    enum: str
    detail: str
    candidate: str
    evidence: Any = None


@dataclass
class SolveContext:
    # Kept for future/debug, but not required for population reporting.
    clue: str
    enumeration: str
    db_answer: str
    stage: str
    assumed_answer: str
    definition_windows: List[str] = field(default_factory=list)
    candidate_list: Optional[List[str]] = None
    hypotheses: Any = None
    raw_stage_output: Any = None


@dataclass
class StageStats:
    entered: int = 0
    solved: int = 0
    forwarded: int = 0
    weak: int = 0
    examples: List[StageExample] = field(default_factory=list)


class CascadeAnalyser:
    """
    Population-level stats container.
    - record_simple(...) is the primary interface for master_solver.
    - record_context(...) is optional and kept for later bespoke debug tooling.
    """

    def __init__(self, max_examples_per_bucket: int = 5):
        self.max_examples_per_bucket = max_examples_per_bucket
        self.stats: Dict[str, StageStats] = {}
        self.stage_order: List[str] = []
        self.contexts: List[SolveContext] = []

    def _ensure(self, stage_name: str) -> StageStats:
        if stage_name not in self.stats:
            self.stats[stage_name] = StageStats()
            self.stage_order.append(stage_name)
        return self.stats[stage_name]

    @staticmethod
    def _pct(part: int, whole: int) -> str:
        if whole <= 0:
            return "0.00%"
        return f"{(100.0 * part / whole):.2f}%"

    def record_simple(
        self,
        stage_name: str,
        entered: int,
        solved: int,
        forwarded: Optional[int] = None,
        weak: int = 0,
        examples: Optional[List[StageExample]] = None,
    ) -> None:
        s = self._ensure(stage_name)
        s.entered += int(entered)
        s.solved += int(solved)
        s.weak += int(weak)
        if forwarded is None:
            # Default: everything not solved is forwarded.
            forwarded = max(0, int(entered) - int(solved))
        s.forwarded += int(forwarded)

        if examples:
            # Keep a small cap, bucket-agnostic for now.
            for ex in examples:
                if len(s.examples) < self.max_examples_per_bucket:
                    s.examples.append(ex)

    def record_context(self, ctx: SolveContext) -> None:
        # Optional: only used if you want to store some examples.
        self.contexts.append(ctx)

    def print_report(self) -> None:
        print("\n=== STAGE BREAKDOWN ===")
        if not self.stage_order:
            print("(no stages recorded)")
            return

        # Baseline is the entered count of the first recorded stage.
        first_stage = self.stage_order[0]
        baseline = self.stats[first_stage].entered

        for name in self.stage_order:
            s = self.stats[name]
            print(
                f"{name}: "
                f"entered={s.entered} "
                f"solved={s.solved} ({self._pct(s.solved, s.entered)}) "
                f"forwarded={s.forwarded} ({self._pct(s.forwarded, s.entered)})"
            )

        print(f"\nBaseline cohort (from '{first_stage}'): {baseline}")
        print("=" * 55)
