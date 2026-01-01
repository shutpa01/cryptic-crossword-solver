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

    total = 0
    matched = 0
    failures = []

    for row in rows:
        clue = row["clue_text"]
        enum = row["enumeration"]
        answer = clean_val(row["answer"])

        ans_norm = norm_letters(answer)
        total_len = parse_enum(enum)

        result = definition_candidates(
            clue_text=clue,
            enumeration=enum,
            graph=graph,
            wordlist=wordlist,
        )

        candidates = [
            c for c in result["candidates"]
            if len(norm_letters(c)) == total_len
        ]

        found = any(norm_letters(c) == ans_norm for c in candidates)

        if found:
            matched += 1
        else:
            failures.append({
                "clue": clue,
                "enumeration": enum,
                "answer": answer,
                "definition_windows": result.get("definition_windows", []),
                "candidates": candidates[:50],
            })

        total += 1

    print("=======================================")
    print(f"Total clues checked:     {total}")
    print(f"Correct candidate hits:  {matched}")
    print(f"Candidate accuracy:      {matched/total*100:.2f}%")
    print("=======================================\n")

    print("===== SAMPLE FAILURES =====")
    for f in random.sample(failures, min(SAMPLE_SIZE, len(failures))):
        print("CLUE:", f["clue"])
        print("ENUM:", f["enumeration"])
        print("ANSWER:", f["answer"])
        print("DEF WINDOWS:", f["definition_windows"])
        print("CANDIDATES:", f["candidates"])
        print("-------------------------------------")


if __name__ == "__main__":
    main()
