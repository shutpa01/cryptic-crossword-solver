import random

from solver.solver_engine.resources import (
    connect_db,
    load_wordlist,
    load_graph,
    clean_val,
    norm_letters,
    parse_enum,
)

from solver.definition.definition_engine_edges import definition_candidates


MAX_CLUES = 5000
SAMPLE_SIZE = 5


def main():
    print("RUNNING MASTER SOLVER")

    conn = connect_db()
    graph = load_graph(conn)
    wordlist = load_wordlist()

    cur = conn.cursor()
    cur.execute("""
        SELECT clue_text, enumeration, answer
        FROM clues
        WHERE enumeration IS NOT NULL AND enumeration != ''
        LIMIT ?
    """, (MAX_CLUES,))
    rows = cur.fetchall()

    # ---- CASCADE COUNTERS ----
    stage1_begin = 0
    stage1_end = 0          # = candidate HIT (answer in candidates)
    stage2_begin = 0        # = stage1 failures
    stage2_end = 0          # = boundary/separator present among failures
    stage3 = 0              # = remaining failures without boundary/separator

    stage3_samples = []

    for row in rows:
        clue = row["clue_text"]
        enum = row["enumeration"]
        answer = clean_val(row["answer"])

        ans_norm = norm_letters(answer)
        total_len = parse_enum(enum)

        stage1_begin += 1

        result = definition_candidates(
            clue_text=clue,
            enumeration=enum,
            graph=graph,
            wordlist=wordlist,
        )

        has_separator = bool(result.get("has_separator", False))

        candidates = [
            c for c in result.get("candidates", [])
            if len(norm_letters(c)) == total_len
        ]

        found = any(norm_letters(c) == ans_norm for c in candidates)

        # ---- STAGE 1 END (HIT) ----
        if found:
            stage1_end += 1
            continue  # parked, removed from cascade

        # ---- STAGE 2 BEGIN (FAILURES) ----
        stage2_begin += 1

        # ---- STAGE 2 END (BOUNDARY AMONG FAILURES) ----
        if has_separator:
            stage2_end += 1
            continue  # parked, removed from cascade

        # ---- STAGE 3 (NO BOUNDARY AMONG FAILURES) ----
        stage3 += 1

        # keep a few for display
        if len(stage3_samples) < 200:
            stage3_samples.append({
                "clue": clue,
                "enumeration": enum,
                "answer": answer,
                "has_separator": has_separator,
                "candidates": candidates[:50],
            })

    def pct(x, denom):
        return (x / denom * 100.0) if denom else 0.0

    print("=======================================")
    print("=== CASCADE BREAKDOWN ===")
    print(f"Stage 1 begin: {stage1_begin}")
    print(f"Stage 1 end:   {stage1_end} ({pct(stage1_end, stage1_begin):.2f}%)")
    print(f"Stage 2 begin: {stage2_begin} ({pct(stage2_begin, stage1_begin):.2f}%)")
    print(f"Stage 2 end:   {stage2_end} ({pct(stage2_end, stage2_begin):.2f}% of Stage 2 begin)")
    print(f"Stage 3:       {stage3} ({pct(stage3, stage2_begin):.2f}% of Stage 2 begin)")
    print("=======================================\n")

    print("===== SAMPLE STAGE 3 CLUES =====")
    for f in random.sample(stage3_samples, min(SAMPLE_SIZE, len(stage3_samples))):
        print("CLUE:", f["clue"])
        print("ENUM:", f["enumeration"])
        print("ANSWER:", f["answer"])
        print("HAS_SEPARATOR:", f["has_separator"])
        print("CANDIDATES:", f["candidates"])
        print("-------------------------------------")


if __name__ == "__main__":
    main()
