from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re
import sqlite3
from collections import defaultdict


# ============================================================
# Data models (BACKWARD COMPATIBLE)
# ============================================================

@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    clue_fragment: str
    produced: str
    used_letters: str
    indicator_support: Optional[str] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class HypothesisResult:
    status: str                     # "pre_solved" | "remaining"
    definition_window: str
    candidate: str
    evidence: Tuple[EvidenceItem, ...]

    # --- legacy fields expected by master_solver ---
    accounted_candidate_letters: int
    candidate_len: int
    used_residual_fragments: Tuple[str, ...]


# ============================================================
# Text helpers
# ============================================================

_WORD_RE = re.compile(r"[A-Za-z]+")


def _tokenize_words(s: str) -> List[str]:
    return _WORD_RE.findall(s.lower())


def _letters_only(s: str) -> str:
    return re.sub(r"[^A-Za-z]", "", s).upper()


def _multiset_counts(s: str) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for ch in s:
        d[ch] = d.get(ch, 0) + 1
    return d


def _is_subset_letters(need: str, have: str) -> bool:
    n = _multiset_counts(need)
    h = _multiset_counts(have)
    for k, v in n.items():
        if h.get(k, 0) < v:
            return False
    return True


def _remove_definition_from_clue(clue: str, definition_window: str) -> str:
    clue_low = clue.lower()
    dw_low = definition_window.lower().strip()
    if not dw_low:
        return clue

    idx = clue_low.find(dw_low)
    if idx < 0:
        return clue

    return (clue[:idx] + " " + clue[idx + len(definition_window):]).strip()


# ============================================================
# DB access (unchanged)
# ============================================================

class HypothesisDB:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (name,),
        )
        return cur.fetchone() is not None

    @staticmethod
    def _table_info(conn: sqlite3.Connection, name: str):
        return list(conn.execute(f"PRAGMA table_info({name});"))

    @staticmethod
    def _pick_col(cols: Sequence[str], *candidates: str) -> Optional[str]:
        s = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in s:
                return s[cand.lower()]
        return None

    def load_indicators(self) -> Dict[str, str]:
        with self._connect() as conn:
            if not self._table_exists(conn, "indicators"):
                return {}

            info = self._table_info(conn, "indicators")
            cols = [r["name"] for r in info]

            col_phrase = self._pick_col(cols, "indicator", "phrase", "text", "word")
            col_op = self._pick_col(cols, "operation", "op", "type")

            if col_phrase is None:
                return {}

            if col_op is None:
                rows = conn.execute(
                    f"SELECT {col_phrase} AS phrase FROM indicators;"
                ).fetchall()
                return {str(r["phrase"]).lower(): "" for r in rows if r["phrase"]}

            rows = conn.execute(
                f"SELECT {col_phrase} AS phrase, {col_op} AS op FROM indicators;"
            ).fetchall()

            out: Dict[str, str] = {}
            for r in rows:
                if r["phrase"]:
                    out[str(r["phrase"]).lower()] = str(r["op"] or "")
            return out

    def load_synonym_map(self) -> Dict[str, List[str]]:
        with self._connect() as conn:
            if not self._table_exists(conn, "synonyms_pairs"):
                return {}

            info = self._table_info(conn, "synonyms_pairs")
            cols = [r["name"] for r in info]

            col_word = self._pick_col(cols, "word", "term")
            col_syn = self._pick_col(cols, "synonym", "syn")

            if col_word is None or col_syn is None:
                return {}

            rows = conn.execute(
                f"SELECT {col_word} AS w, {col_syn} AS s FROM synonyms_pairs;"
            ).fetchall()

            mp: Dict[str, List[str]] = defaultdict(list)
            for r in rows:
                if r["w"] and r["s"]:
                    w = str(r["w"]).lower()
                    s = str(r["s"]).lower()
                    mp[w].append(s.upper())
                    mp[s].append(w.upper())
            return mp


# ============================================================
# Hypothesis stage (SPEC-ALIGNED, COMPATIBLE)
# ============================================================

class HypothesisStage:
    def __init__(self, db_path: str):
        self.db = HypothesisDB(db_path)
        self.indicators = self.db.load_indicators()
        self.syn_map = self.db.load_synonym_map()

    def _indicator_support(self, clue: str) -> Optional[str]:
        clue_l = clue.lower()
        for ind, op in self.indicators.items():
            if ind in clue_l:
                return op or ind
        return None

    def evaluate_candidate_for_window(
        self, clue: str, definition_window: str, candidate: str
    ) -> HypothesisResult:

        residual = _remove_definition_from_clue(clue, definition_window)
        unused_words = _tokenize_words(residual)
        unused_letters = list(_letters_only(candidate))

        evidence: List[EvidenceItem] = []
        used_fragments: List[str] = []

        progress = True
        while progress:
            progress = False
            cand_letters = "".join(unused_letters)

            for w in list(unused_words):
                for syn in self.syn_map.get(w, []):
                    letters = _letters_only(syn)
                    if letters and _is_subset_letters(letters, cand_letters):
                        ev = EvidenceItem(
                            kind="synonym",
                            clue_fragment=w,
                            produced=syn,
                            used_letters=letters,
                            indicator_support=self._indicator_support(clue),
                        )
                        evidence.append(ev)
                        used_fragments.append(w)
                        progress = True

                        for ch in letters:
                            if ch in unused_letters:
                                unused_letters.remove(ch)
                        unused_words.remove(w)
                        break
                if progress:
                    break

        meaningful_leftover = [w for w in unused_words if len(w) >= 3]

        status = (
            "pre_solved"
            if not unused_letters and not meaningful_leftover and evidence
            else "remaining"
        )

        accounted = len(_letters_only(candidate)) - len(unused_letters)

        return HypothesisResult(
            status=status,
            definition_window=definition_window,
            candidate=candidate,
            evidence=tuple(evidence),
            accounted_candidate_letters=accounted,
            candidate_len=len(_letters_only(candidate)),
            used_residual_fragments=tuple(used_fragments),
        )

    def run_for_clue(
        self,
        clue: str,
        definition_windows: Sequence[str],
        candidates_by_window: Dict[str, Sequence[str]],
    ) -> HypothesisResult:

        for dw in definition_windows:
            for cand in candidates_by_window.get(dw, []):
                res = self.evaluate_candidate_for_window(clue, dw, cand)
                if res.status == "pre_solved":
                    return res

        return HypothesisResult(
            status="remaining",
            definition_window=definition_windows[0] if definition_windows else "",
            candidate="",
            evidence=tuple(),
            accounted_candidate_letters=0,
            candidate_len=0,
            used_residual_fragments=tuple(),
        )
