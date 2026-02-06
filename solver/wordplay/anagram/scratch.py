from solver.wordplay.anagram.compound_wordplay_analyzer import CompoundWordplayAnalyzer

cwa = CompoundWordplayAnalyzer("path/to/your/cryptic.db")

# Check indicator lookup
ind = cwa.db.lookup_indicator("endless")
print(f"endless indicator: {ind}")

# Check synonym lookup
subs = cwa.db.lookup_substitution("dads", max_synonym_length=5)
print(f"dads substitutions: {subs}")