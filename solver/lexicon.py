import sqlite3
from collections import defaultdict

DB_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic.db"


class Lexicon:
    """
    Central access point for all cryptic crossword lexical knowledge.
    Loads indicators, synonyms, substitutions, and wordplay rules
    from the live SQLite DB and exposes access/update methods.
    """

    def __init__(self):

        # Path to the main lexicon database
        self.db_path = DB_PATH

        # Core lexicon structures
        self.synonyms = defaultdict(set)
        self.substitutions = defaultdict(set)
        self.variants = defaultdict(set)

        # MUST be sets — loader uses .add()
        self.indicators = defaultdict(set)

        self.wordplay_rules = []

        # Lookup dictionaries
        self.indicator_lookup = {}
        self.synonym_lookup = {}
        self.variant_lookup = {}
        self.substitution_lookup = {}

        # Index of valid words by length
        self.words_by_length = {}

        # Load everything
        self.reload()

    # ------------------------------------------------------------
    # DATABASE ACCESS
    # ------------------------------------------------------------

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------
    # RELOADING ALL TABLES
    # ------------------------------------------------------------

    def reload(self):
        self._load_indicators()
        self._load_synonyms()
        self._load_substitutions()
        self._load_wordplay()
        self._build_word_index()

    # ------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------

    def _load_indicators(self):
        self.indicators.clear()
        self.indicator_lookup.clear()

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT word, wordplay_type FROM indicators")
            rows = cur.fetchall()

        for word, wtype in rows:
            w = word.strip().lower()
            t = wtype.strip().lower()
            self.indicators[t].add(w)
            self.indicator_lookup[w] = t

    def _load_synonyms(self):
        self.synonyms.clear()

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT word, synonyms FROM synonyms")
            rows = cur.fetchall()

        # Your DB stores rows like:
        # word | synonyms   (where synonyms is a string)
        # You attempted to treat both sides as bidirectional: good idea.
        for word, syn_str in rows:
            if not syn_str:
                continue

            base = word.strip().lower()

            # normalise formats like:
            # "['foo','bar']" or "foo,bar"
            syn_list = [
                s.strip().lower()
                for s in syn_str.replace("[", "").replace("]", "").replace("'", "").split(",")
                if s.strip()
            ]

            for syn in syn_list:
                self.synonyms[base].add(syn)
                self.synonyms[syn].add(base)  # bidirectional

    def _load_substitutions(self):
        self.substitutions.clear()

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT original_word, substitution FROM substitutions")
            rows = cur.fetchall()

        for w, sub in rows:
            w = w.strip().lower()
            sub = sub.strip().lower()
            self.substitutions[w].add(sub)

    def _load_wordplay(self):
        self.wordplay_rules.clear()

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM wordplay")
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()

        for row in rows:
            self.wordplay_rules.append(dict(zip(cols, row)))

    # ------------------------------------------------------------
    # QUERY METHODS
    # ------------------------------------------------------------

    def is_indicator(self, word):
        return word.lower() in self.indicator_lookup

    def get_indicator_type(self, word):
        return self.indicator_lookup.get(word.lower())

    def get_synonyms(self, word):
        return list(self.synonyms.get(word.lower(), []))

    def get_substitutions(self, word):
        return list(self.substitutions.get(word.lower(), []))

    def find_matching_indicators(self, clue_text):
        words = [w.strip(".,?!;:").lower() for w in clue_text.split()]
        return [(w, self.get_indicator_type(w)) for w in words if w in self.indicator_lookup]

    # ------------------------------------------------------------
    # UPDATE METHODS
    # ------------------------------------------------------------

    def add_indicator(self, word, wtype):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO indicators (word, wordplay_type) VALUES (?, ?)",
                (word.lower(), wtype.lower()),
            )
            conn.commit()
        self._load_indicators()

    def add_synonym(self, a, b):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO synonyms (word, synonyms) VALUES (?, ?)",
                (a.lower(), b.lower()),
            )
            cur.execute(
                "INSERT INTO synonyms (word, synonyms) VALUES (?, ?)",
                (b.lower(), a.lower()),
            )
            conn.commit()
        self._load_synonyms()

    def add_substitution(self, original, sub):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO substitutions (original_word, substitution) VALUES (?, ?)",
                (original.lower(), sub.lower()),
            )
            conn.commit()
        self._load_substitutions()

    # ------------------------------------------------------------
    # WORD INDEX
    # ------------------------------------------------------------

    def _build_word_index(self):
        self.words_by_length = {}
        wordset = set()

        # Add synonym keys and values
        for w, syns in self.synonyms.items():
            wordset.add(w)
            for s in syns:
                wordset.add(s)

        # Add substitution outputs
        for w, subs in self.substitutions.items():
            wordset.add(w)
            for s in subs:
                wordset.add(s)

        # Add variants
        for w, vars in self.variants.items():
            wordset.add(w)
            for v in vars:
                wordset.add(v)

        # Build length-indexed map
        for w in wordset:
            l = len(w)
            if l not in self.words_by_length:
                self.words_by_length[l] = []
            self.words_by_length[l].append(w)

    # ------------------------------------------------------------
    # WORDPLAY RULE ACCESS
    # ------------------------------------------------------------

    def get_wordplay_rules(self):
        return self.wordplay_rules

    # ------------------------------------------------------------
    # DEBUG / INTROSPECTION
    # ------------------------------------------------------------

    def summary(self):
        return {
            "indicators": sum(len(v) for v in self.indicators.values()),
            "synonyms": len(self.synonyms),
            "substitutions": len(self.substitutions),
            "wordplay_rules": len(self.wordplay_rules),
        }


# Singleton instance
LEXICON = Lexicon()
