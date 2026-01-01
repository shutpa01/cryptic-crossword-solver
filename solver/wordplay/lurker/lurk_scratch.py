from solver.solver_engine.resources import (
    connect_db,
    clean_val,
    norm_letters,
    parse_enum,
)

from solver.wordplay.lurker.lurker_stage import generate_lurker_hypotheses


MAX_ROWS = 500
SAMPLE_SIZE = 10


def main():
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT clue_text, enumeration, answer
        FROM clues
        WHERE wordplay_type = 'hidden'
        AND enumeration IS NOT NULL
        LIMIT ?
    """, (MAX_ROWS,))

    rows = cur.fetchall()

    tested = 0
    detected = 0
    successes = []
    failures = []

    for row in rows:
        clue = row["clue_text"]
        enum = row["enumeration"]
        answer = clean_val(row["answer"])

        total_len = parse_enum(enum)
        ans_norm = norm_letters(answer)

        hypotheses = generate_lurker_hypotheses(
            clue_text=clue,
            enumeration=total_len,
            candidates=[answer],   # force true answer as candidate
        )

        found = any(
            norm_letters(h["answer"]) == ans_norm
            for h in hypotheses
        )

        tested += 1

        if found:
            detected += 1
            if len(successes) < SAMPLE_SIZE:
                successes.append({
                    "clue": clue,
                    "enum": enum,
                    "answer": answer,
                })
        else:
            if len(failures) < SAMPLE_SIZE:
                failures.append({
                    "clue": clue,
                    "enum": enum,
                    "answer": answer,
                })

    print("=======================================")
    print("=== LURKER DIAGNOSTIC TEST ===")
    print(f"Tested:   {tested}")
    print(f"Detected: {detected}")
    print(f"Missed:   {tested - detected}")
    if tested:
        print(f"Hit rate: {detected / tested * 100:.2f}%")
    print("=======================================\n")

    if successes:
        print("=== SAMPLE SUCCESSES ===")
        for s in successes:
            print("CLUE:", s["clue"])
            print("ENUM:", s["enum"])
            print("ANSWER:", s["answer"])
            print("-------------------------------------")

    if failures:
        print("\n=== SAMPLE FAILURES ===")
        for f in failures:
            print("CLUE:", f["clue"])
            print("ENUM:", f["enum"])
            print("ANSWER:", f["answer"])
            print("-------------------------------------")


if __name__ == "__main__":
    main()
