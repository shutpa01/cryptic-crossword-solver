from solver.wordplay.anagram.anagram_stage import EVIDENCE_SYSTEM_AVAILABLE, generate_anagram_hypotheses

print(f"EVIDENCE_SYSTEM_AVAILABLE: {EVIDENCE_SYSTEM_AVAILABLE}")

clue = "Hobby that's hip, possibly in the recent past (9)"
candidates = ['PHILATELY']  # Assume this is in definition candidates

result = generate_anagram_hypotheses(clue, 9, candidates)
print(f"Hypotheses returned: {len(result)}")
for h in result:
    print(f"  - {h.get('answer')}: {h.get('solve_type')}, {h.get('evidence_type')}")