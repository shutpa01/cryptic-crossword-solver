# import sqlite3
# import re
# import random
#
# DB_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic.db"
# WORDLIST_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\wordlist.txt"
#
# MAX_CLUES = 5000
# MAX_DEF_WORDS = 6
#
# STOP_DEFS = {
#     "a","an","the","it","its","this","that","these","those",
#     "he","she","they","them","his","her","their",
#     "one","ones","some","any","each","all","both"
# }
#
# # ------------------------- NORMALISATION -------------------------
#
# def clean_key(x: str) -> str:
#     if not x:
#         return ""
#     x = x.strip().lower()
#     x = re.sub(r"^[^a-z]+", "", x)
#     x = re.sub(r"[^a-z]+$", "", x)
#     return x
#
# def clean_val(x: str) -> str:
#     if not x:
#         return ""
#     x = x.strip()
#     x = re.sub(r"^[^A-Za-z]+", "", x)
#     x = re.sub(r"[^A-Za-z]+$", "", x)
#     return x
#
# def norm_letters(s: str) -> str:
#     return re.sub(r"[^A-Za-z]", "", s or "").lower()
#
# # ------------------------- ENUM + WORDLIST -------------------------
#
# def parse_enum(en):
#     return sum(map(int, re.findall(r"\d+", en or "")))
#
# def tokenize(s):
#     return [re.sub(r"^[^\w]+|[^\w]+$", "", w)
#             for w in (s or "").split() if w]
#
# def load_wordlist():
#     try:
#         with open(WORDLIST_PATH, encoding="utf-8") as f:
#             return [w.strip() for w in f if w.strip()]
#     except:
#         return []
#
# wordlist = load_wordlist()
#
# def matches_enumeration(word, enumeration):
#     parts = list(map(int, re.findall(r"\d+", enumeration or "")))
#     cleaned = re.sub(r"[^A-Za-z ]", "", word or "")
#     chunks = cleaned.split()
#     if len(chunks) != len(parts):
#         return False
#     return all(len(chunks[i]) == parts[i] for i in range(len(parts)))
#
# def enumeration_candidates(enum):
#     return [w for w in wordlist if matches_enumeration(w, enum)]
#
# # ------------------------- DEF ROOTS -------------------------
#
# def normalise_definition_root(dp):
#     out = {dp}
#     if dp.endswith("'s") or dp.endswith("’s"):
#         out.add(dp[:-2])
#     if dp.endswith("s") and len(dp) > 3:
#         out.add(dp[:-1])
#     return list(out)
#
# # ---------------------- BIDIRECTIONAL GRAPH ---------------------
#
# def add_pair(graph, a, b):
#     ak = clean_key(a)
#     bv = clean_val(b)
#     if ak and bv:
#         graph.setdefault(ak, set()).add(bv)
#
#     bk = clean_key(b)
#     av = clean_val(a)
#     if bk and av:
#         graph.setdefault(bk, set()).add(av)
#
# def load_graph(conn):
#     cur = conn.cursor()
#     G = {}
#
#     cur.execute("SELECT phrase, target FROM definition_mappings")
#     for phrase, target in cur.fetchall():
#         add_pair(G, phrase, target)
#
#     cur.execute("SELECT definition, answer FROM definition_answers")
#     for d, ans in cur.fetchall():
#         add_pair(G, d, ans)
#
#     cur.execute("SELECT word, synonym FROM synonyms_pairs")
#     for w, s in cur.fetchall():
#         add_pair(G, w, s)
#
#     return {k: list(v) for k, v in G.items()}
#
# def get_candidates(def_phrase, graph):
#     key = clean_key(def_phrase)
#     if not key:
#         return []
#     return graph.get(key, [])
#
# # ------------------------------ MAIN ------------------------------
#
# def main():
#     print("RUNNING — CLEAN 75% VERSION (NO OVERLAP FILTER, NO EMBEDDINGS)")
#
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     graph = load_graph(conn)
#
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT id, clue_text, enumeration, answer
#         FROM clues
#         WHERE enumeration IS NOT NULL AND enumeration != ''
#         LIMIT ?
#     """, (MAX_CLUES,))
#     rows = cur.fetchall()
#
#     total = 0
#     matched = 0
#     failures = []
#
#     for row in rows:
#         clue_text = row["clue_text"]
#         answer_raw = row["answer"]
#         enum = row["enumeration"]
#         answer = clean_val(answer_raw)
#         ans_norm = norm_letters(answer)
#
#         total_len = parse_enum(enum)
#
#         body = re.sub(r"\s*\([^)]*\)\s*$", "", clue_text or "").strip()
#         tokens = tokenize(body)
#         if not tokens:
#             continue
#
#         def_windows = []
#
#         for k in range(1, min(MAX_DEF_WORDS, len(tokens))+1):
#             dp = clean_key(" ".join(tokens[:k]))
#             if dp and dp not in STOP_DEFS:
#                 def_windows.append(dp)
#
#         for k in range(1, min(MAX_DEF_WORDS, len(tokens))+1):
#             dp = clean_key(" ".join(tokens[-k:]))
#             if dp and dp not in STOP_DEFS:
#                 def_windows.append(dp)
#
#         all_candidates = []
#
#         for dp in def_windows:
#             for root in normalise_definition_root(dp):
#                 all_candidates.extend(get_candidates(root, graph))
#
#         all_candidates.extend(enumeration_candidates(enum))
#
#         length_ok = [c for c in all_candidates
#                      if len(norm_letters(c)) == total_len]
#
#         found = any(norm_letters(c) == ans_norm for c in length_ok)
#
#         if found:
#             matched += 1
#         else:
#             failures.append({
#                 "clue": clue_text,
#                 "answer": answer,
#                 "enumeration": enum,
#                 "definition_windows": def_windows,
#                 "candidates": length_ok[:50],
#             })
#
#         total += 1
#
#     print("=======================================")
#     print(f"Total clues checked:     {total}")
#     print(f"Correct candidate hits:  {matched}")
#     print(f"Candidate accuracy:      {matched/total*100:.2f}%")
#     print("=======================================\n")
#
#     sample = random.sample(failures, min(10, len(failures)))
#     print("===== SAMPLE FAILURES =====")
#     for f in sample:
#         print("CLUE:", f["clue"])
#         print("ANSWER:", f["answer"])
#         print("DEF WINDOWS:", f["definition_windows"])
#         print("CANDIDATES:", f["candidates"])
#         print("-------------------------------------")
#
# if __name__ == "__main__":
#     main()
