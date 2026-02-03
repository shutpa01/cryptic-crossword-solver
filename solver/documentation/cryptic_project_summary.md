# Cryptic Crossword Explanation System - Project Summary
## Date: February 2025 (Updated 2 Feb 2025)

---

## PROJECT GOAL

Build a system that **explains** how cryptic crossword clues work, NOT solves them from scratch.

**Key insight:** We always have the answer (from scraped puzzle data). So instead of generating candidates, we explain: "Given this clue and this known answer, how does the wordplay construct the answer?"

---

## ARCHITECTURAL DECISION: USE_KNOWN_ANSWER = True

In `pipeline_simulator.py` line 59:
```python
USE_KNOWN_ANSWER = True  # Answer becomes the only candidate
```

This changes everything:
- No need to generate/rank candidate answers
- Each detection stage asks: "Does this mechanism explain how clue → known answer?"
- Much higher accuracy because we're not guessing

---

## CASCADE PIPELINE

Each stage receives unsolved clues, tries its detection method, marks solved/unsolved, persists results, passes unsolved forward.

```
Raw Cohort (stage_input)
    ↓
DD Detection (stage_dd) → solved exits
    ↓
Lurker Detection (stage_lurker) → solved exits  
    ↓
Exact Anagram (stage_anagram) → solved exits
    ↓
Compound Anagram (stage_compound) → solved/partial exits
    ↓
General Wordplay (stage_general) → NEXT STAGE NEEDED
```

---

## THREE-STATE RESOLUTION FLAG (NEW - 2 Feb 2025)

The `fully_resolved` field in stage_compound now uses three states:

| Value | Meaning | Action |
|-------|---------|--------|
| 0 | No anagram component found | Forward to general wordplay stage |
| 1 | Fully resolved | Done - complete explanation |
| 2 | Partial anagram | Stay in anagram cohort - has valid indicator + fodder but needs more letters |

**Why this matters:** Partial anagrams like CAMOUFLAGE (anagram of GUACAMOLE + F) shouldn't go to the general stage - they're anagram clues that just need compound analysis to find the extra letters.

---

## DATABASE: pipeline_stages.db

Location: `C:\Users\shute\PycharmProjects\cryptic_solver\data\pipeline_stages.db`

Tables:
- `pipeline_meta` - run tracking
- `stage_input` - raw cohort (clue_id, clue_text, answer, enumeration, source, puzzle_number, wordplay_type)
- `stage_dd` - double definition results (hit_found, matched_answer, windows)
- `stage_lurker` - hidden word results (hit_found, lurker_answer, container_text)
- `stage_anagram` - anagram results (hit_found, fodder_words, fodder_letters, solve_type, unused_words)
- `stage_evidence` - evidence scoring
- `stage_compound` - compound wordplay (formula, quality, definition_window, substitutions, fully_resolved, unresolved_words)
- `stage_definition` - definition matching
- `stage_definition_failed` - clues that failed definition gate

Key queries:
```sql
-- Clues to forward to general stage
SELECT * FROM stage_compound WHERE fully_resolved = 0

-- Partial anagrams (stay in anagram cohort)
SELECT * FROM stage_compound WHERE fully_resolved = 2

-- Fully resolved
SELECT * FROM stage_compound WHERE fully_resolved = 1
```

---

## DATABASE: cryptic_new.db

Location: `C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db`

Key tables:
- `clues` - source clue data
- `definition_answers_augmented` - known definition→answer mappings (e.g., "Quality" → "ATTRIBUTE")
- `indicators` - wordplay indicator words by type (anagram, reversal, hidden, homophone, acrostic, deletion, container, insertion)
- `wordplay` - substitution rules (e.g., "book" → "B", "knight" → "N")
- `synonyms_pairs` - synonym lookup (e.g., "don" → "wear")
- `homophones` - homophone pairs (e.g., "new" → "knew")

---

## KEY FILES

### pipeline_simulator.py
- Main orchestrator for DD → Definition → Anagram → Lurker stages
- `USE_KNOWN_ANSWER = True` makes answer the only candidate
- Persists all stage results to pipeline_stages.db

### pipeline_persistence.py
- Handles saving results to pipeline_stages.db
- **FIXED (2 Feb 2025):** Line 510 now preserves numeric values for fully_resolved

### anagram_analysis.py  
- Calls pipeline_simulator
- Filters for clues with anagram hits
- Runs compound analysis via ExplanationSystemBuilder
- **GAP:** Clues WITHOUT anagram hits are dropped (line 62-67)

### anagram_evidence_system.py
- Sophisticated anagram detection
- Handles exact, partial, deletion, insertion cases
- Enforces fodder rules: proximity, contiguity, whole words
- **BUG IDENTIFIED (2 Feb 2025):** Accepts invalid fodder like enumerations ("5", "7") - see fix below

### compound_wordplay_analyzer.py
- Explains compound wordplay (anagram + substitutions)
- Database lookups for indicators and substitutions
- Builds formulas like: "anagram(NORMA) + DO (party) = DOORMAN"
- **FIXED (2 Feb 2025):** Definition window now uses database lookup as fallback
- **FIXED (2 Feb 2025):** Partial anagram check now uses `solve_type == 'anagram_evidence_partial'`

### explanation_builder.py
- Presentation layer for explanations
- Quality assessment (solved/high/medium/low/none)
- Formula and breakdown formatting

---

## COMPLETED FIXES (2 Feb 2025)

### 1. Definition Window Database Lookup
**Problem:** Definition finding failed when graph lookup returned empty.

**Solution:** Added `_find_definition_from_db()` method that queries `definition_answers_augmented` directly as fallback.

### 2. Pipeline Persistence - Preserve Value 2
**Problem:** Line 510 used `1 if compound.get('fully_resolved') else 0` which converted value 2 to 1.

**Solution:** Changed to `compound.get('fully_resolved', 0)` to preserve actual numeric value.

**File:** `pipeline_persistence.py`
```python
# Line 510 - BEFORE (buggy):
1 if compound.get('fully_resolved') else 0,

# Line 510 - AFTER (fixed):
compound.get('fully_resolved', 0),
```

### 3. Partial Anagram Detection in Compound Stage
**Problem:** Original check `if compound_solution and anagram_indicator and fodder_words` triggered for clues with invalid fodder (enumerations like "5") because `fodder_words = ["5"]` is truthy.

**Solution:** Check if anagram stage already validated as partial anagram by looking for `solve_type == 'anagram_evidence_partial'`.

**File:** `compound_wordplay_analyzer.py` (two locations: ~line 676 and ~line 949)
```python
# Check if anagram stage already validated this as a partial anagram
anagrams = case.get('anagrams', [])
is_validated_partial = any(
    hit.get('solve_type') == 'anagram_evidence_partial' 
    for hit in anagrams
)
if compound_solution and is_validated_partial:
    if compound_solution.get('letters_still_needed'):
        compound_solution['fully_resolved'] = 2
```

---

## OUTSTANDING BUG: FALSE PARTIAL ANAGRAMS IN ANAGRAM STAGE

### The Problem

The anagram stage (`anagram_evidence_system.py`) is incorrectly marking clues as `solve_type = 'anagram_evidence_partial'` when the fodder is invalid:

| Clue | Answer | fodder_words | fodder_letters | Problem |
|------|--------|--------------|----------------|---------|
| Mare rears up... | MEAD | ["4"] | "" | Enumeration as fodder |
| Son's removed... | ICILY | ["5"] | "" | Enumeration as fodder |
| League accountant... | CAHOOTS | ["7"] | "" | Enumeration as fodder |
| Daughter to take... | ROAD MAP | ["—"] | "" | Punctuation as fodder |

These have `fodder_letters = ""` (empty) but still get `solve_type = 'anagram_evidence_partial'`.

### Where the Bug Is

In `anagram_evidence_system.py`, find where `solve_type` is set to `'anagram_evidence_partial'`. The condition that allows this needs to be tightened.

### The Fix Required

Before setting `solve_type = 'anagram_evidence_partial'`, validate:

1. **fodder_letters must contain alphabetic content:**
```python
fodder_alpha = ''.join(c for c in fodder_letters if c.isalpha())
if len(fodder_alpha) == 0:
    # NOT a valid partial anagram - fodder has no letters
    # Do not set solve_type = 'anagram_evidence_partial'
```

2. **All fodder letters must appear in the answer** (standard anagram criterion):
```python
answer_letters = list(answer.upper())
for c in fodder_alpha.upper():
    if c in answer_letters:
        answer_letters.remove(c)
    else:
        # Fodder contains letter not in answer - invalid
        break
```

### How to Find the Code Location

Search `anagram_evidence_system.py` for:
```bash
grep -n "anagram_evidence_partial" inactive_anagram_evidence_system.py
```

This will show where the solve_type is assigned. Add the validation checks before that assignment.

### Expected Outcome After Fix

| Clue | fodder_words | fodder_letters | Valid? | solve_type |
|------|--------------|----------------|--------|------------|
| MEAD | ["4"] | "" | NO - no letters | NOT partial |
| ICILY | ["5"] | "" | NO - no letters | NOT partial |
| CAMOUFLAGE | ["guacamole"] | "guacamole" | YES - all letters in answer | anagram_evidence_partial |
| WASHBOARD | ["was", "hard"] | "washard" | YES - all letters in answer | anagram_evidence_partial |

---

## NEXT STEPS

1. **Fix anagram_evidence_system.py** - Add fodder validation before setting `solve_type = 'anagram_evidence_partial'`

2. **Re-run pipeline** - Verify false positives are eliminated

3. **Build general wordplay stage** - For clues with `fully_resolved = 0`:
   - Homophones
   - Acrostics  
   - Simple reversals
   - Deletions
   - Alternate letters

---

## FILE LOCATIONS

```
C:\Users\shute\PycharmProjects\cryptic_solver\
├── data\
│   ├── cryptic_new.db              # Main database
│   └── pipeline_stages.db          # Stage results
├── solver\
│   ├── solver_engine\
│   │   ├── pipeline_simulator.py   # Main orchestrator
│   │   ├── pipeline_persistence.py # DB persistence (FIXED)
│   │   └── evidence_analysis.py
│   └── wordplay\
│       └── anagram\
│           ├── anagram_analysis.py
│           ├── anagram_evidence_system.py  # NEEDS FIX
│           ├── compound_wordplay_analyzer.py  # FIXED
│           └── explanation_builder.py
```

---

## KEY PRINCIPLES

**"Absolute sanctity of working systems"**

Don't modify working code. Instead:
- Create new files that consume output of working stages
- Add new stages to the cascade
- Make surgical, minimal patches when absolutely necessary
- Always backup before changes

**Validation belongs at the source**

The anagram stage should validate fodder before marking as partial anagram. Downstream stages (compound) should trust upstream validation, not re-implement it.
