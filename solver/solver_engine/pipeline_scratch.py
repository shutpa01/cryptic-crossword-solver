from solver.solver_engine.resources import load_graph, connect_db, norm_letters
from solver.definition.definition_engine_edges import definition_candidates
from pipeline_simulator import _length_filter

conn = connect_db()
graph = load_graph(conn)

clue = "Rhythm of dance organised with church (7)"
enum = "(7)"

def_result = definition_candidates(clue, enum, graph)
raw_candidates = def_result.get("candidates", [])

print(f"Raw candidates count: {len(raw_candidates)}")
print(f"CADENCE in raw: {'CADENCE' in raw_candidates}")
print(f"cadence in raw: {'cadence' in raw_candidates}")

filtered = _length_filter(raw_candidates, enum)
print(f"Filtered candidates count: {len(filtered)}")
print(f"CADENCE in filtered: {'CADENCE' in filtered}")
print(f"cadence in filtered: {'cadence' in filtered}")