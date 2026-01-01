# Database Schema: cryptic.db

**Generated:** 2025-12-11 07:57:33

---

## Tables Overview

Total tables: 16

- **clues**: 327,189 rows
- **clues_fts**: 327,189 rows
- **clues_fts_config**: 1 rows
- **clues_fts_data**: 6,824 rows
- **clues_fts_docsize**: 327,189 rows
- **clues_fts_idx**: 6,263 rows
- **definition_answers**: 162,185 rows
- **definition_mappings**: 57,289 rows
- **indicator_type_map**: 0 rows
- **indicators**: 18,496 rows
- **sqlite_sequence**: 6 rows
- **substitutions**: 7 rows
- **successful_patterns**: 5 rows
- **synonyms**: 78,669 rows
- **synonyms_pairs**: 606,400 rows
- **wordplay**: 771 rows

---

## Table: `clues`

**Row count:** 327,189

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `source` | TEXT | NOT NULL |
| `puzzle_number` | TEXT | - |
| `publication_date` | DATE | - |
| `clue_number` | TEXT | NOT NULL |
| `direction` | TEXT | NOT NULL |
| `clue_text` | TEXT | NOT NULL |
| `enumeration` | TEXT | - |
| `answer` | TEXT | NOT NULL |
| `definition` | TEXT | - |
| `explanation` | TEXT | - |
| `ai_explanation` | TEXT | - |
| `wordplay_type` | TEXT | - |

### Indexes

- `idx_clues_date`: (publication_date)
- `idx_clues_puzzle`: (puzzle_number)
- `idx_clues_answer`: (answer)
- `idx_clues_source`: (source)

### Sample Data

```
{'id': 1, 'source': 'telegraph', 'puzzle_number': '31093', 'publication_date': '2025-11-25', 'clue_number': '1a', 'direction': 'across', 'clue_text': 'Vehicle moved slowly and aimlessly (5)', 'enumeration': '5', 'answer': 'MOPED', 'definition': 'double-definition', 'explanation': 'a double definition; a motorised two-wheeler & gloomy/broody behaviour.', 'ai_explanation': None, 'wordplay_type': 'double_definition'}
{'id': 2, 'source': 'telegraph', 'puzzle_number': '31093', 'publication_date': '2025-11-25', 'clue_number': '4a', 'direction': 'across', 'clue_text': 'Decorated sailor following the Spanish lecture (9)', 'enumeration': '9', 'answer': 'ELABORATE', 'definition': 'Decorated', 'explanation': 'the in Spanish + the usual sailor/merchant ship rank abbreviation + a synonym for lecture.', 'ai_explanation': None, 'wordplay_type': None}
{'id': 3, 'source': 'telegraph', 'puzzle_number': '31093', 'publication_date': '2025-11-25', 'clue_number': '9a', 'direction': 'across', 'clue_text': 'Met doctor as I shifted hospital’s shed (9)', 'enumeration': '9', 'answer': 'SATISFIED', 'definition': 'Met', 'explanation': 'an anagram (doctor) of AS I S[h]IFTED excluding (shed) the single letter for\xa0Hospital.', 'ai_explanation': None, 'wordplay_type': 'anagram'}
```

---

## Table: `clues_fts`

**Row count:** 327,189

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `clue_text` | ANY | - |
| `answer` | ANY | - |
| `definition` | ANY | - |
| `explanation` | ANY | - |

### Sample Data

```
{'clue_text': 'Vehicle moved slowly and aimlessly (5)', 'answer': 'MOPED', 'definition': 'double-definition', 'explanation': 'a double definition; a motorised two-wheeler & gloomy/broody behaviour.'}
{'clue_text': 'Decorated sailor following the Spanish lecture (9)', 'answer': 'ELABORATE', 'definition': 'Decorated', 'explanation': 'the in Spanish + the usual sailor/merchant ship rank abbreviation + a synonym for lecture.'}
{'clue_text': 'Met doctor as I shifted hospital’s shed (9)', 'answer': 'SATISFIED', 'definition': 'Met', 'explanation': 'an anagram (doctor) of AS I S[h]IFTED excluding (shed) the single letter for\xa0Hospital.'}
```

---

## Table: `clues_fts_config`

**Row count:** 1

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `k` | ANY | PRIMARY KEY, NOT NULL |
| `v` | ANY | - |

### Indexes

- `sqlite_autoindex_clues_fts_config_1`: UNIQUE (k)

### Sample Data

```
{'k': 'version', 'v': 4}
```

---

## Table: `clues_fts_data`

**Row count:** 6,824

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `block` | BLOB | - |

### Sample Data

```
{'id': 1, 'block': b'\x93\xfc\x15\x81\xb6\xd4v\x96\xe3N\x98\xdfw\x81\xa3\xd4\x03'}
{'id': 10, 'block': b"\x00\x00\x00\x00\x04\x0b\xbd\n\x00\x03\x05\x01r\x07\x01t\x08\x01\x1f\x00\x03\r\x01\x82&\t\x01\x82'\x0b\x01\x82\x18\x04\x04\x01\x86t\x8dI\x02\x862\x8c1\x03\x85\x0c\x89s\x04\x84\x11\x87k\x00\x01\x06\x01\x95k"}
{'id': 137438954356, 'block': b'\x00\x00\x06\xa2\x080learned\x96\x17\x02\x06\x8a\x15\x02\x08\xa3/\x08\x04\x01\x03\x07\x8c\x0e\x06\x01\x03\x1c\x86M\x06\x01\x01\x02\x90q\x02\x05\x8bo\x08\x04\x01\x03\x0c\x88J\x06\x01\x01\x02\x8e\x11\x08\x02\x01\x02\x02\x82C\x08\x02\x01\x02\x02\x8d\x12\x08\x02\x01\x02\x02\x81\n\x08\x01\x03\x06(\x85\\\x06\x01\x03\x08\x8bT\x06\x01\x03\x13\x8a\x00\x08\x02\x01\x02\x02.\x08\x02\x01\x03\x05\x91\x03\x02\x02\x82\x0c\x06\x01\x03I\x81w\x06\x01\x01\x02\x87J\x08\x02\x01\x02\x022\x08\x06\x01\x03\x12\x9e[\x06\x01\x03\x0e\x84\x1b\x08\x02\x01\x03\x05\x90C\x06\x01\x03\x05\x89V\x08\x02\x01\x02\x02\xa5a\x0e\x08\x01\x02\x08\x01\x03\x04\x86R\x08\x02\x01\x02\x02\x834\x06\x01\x03\n\x84.\x08\x04\x01\x02\x04\x9dI\x08\x02\x01\x02\x02\x83w\x02\n\x83?\x02\x04\x84~\x08\x02\x01\x02\x02\x82$\x06\x01\x03\x0e\x87Z\x08\x05\x01\x02\x05\x97d\x02\x04\x9cu\x06\x01\x03\t\x08\x04ness\x82\x88!\x06\x01\x03\x07\x07\x01rQ\x06\x01\x03\x18;\x06\x01\x03\x0b\x83\x1e\x06\x01\x03\x0e\x81\r\x06\x01\x03\x1c\x8bG\x06\x01\x03\x07\x81\x1b\x08\x02\x01\x02\x02\x85]\x06\x01\x03\x1d\x87X\x06\x01\x03\x13\x82\x1a\x06\x01\x03\x0f\x89J\x08\x02\x01\x03\x07h\x08\x02\x01\x03\x08=\x06\x01\x03\x10\x8cv\x06\x01\x03\x1e\x81\x1e\x08\x02\x01\x03\x06*\x08\x04\x01\x03\x13\x1e\x06\x01\x03\x0c\x813\x06\x01\x03\x1cq\x06\x01\x03\x11\x88G\x06\x01\x03\ta\x06\x01\x01\x02\x826\x08\x05\x01\x03\x06\x82e\x06\x01\x03\x1b~\x06\x01\x03\x11\x81\x16\x06\x01\x03\t\x83[\x06\x01\x03\x18h\x06\x01\x01\x02\x1e\x06\x01\x03\x0e\x83\'\x06\x01\x03\t\x81y\x08\x02\x01\x03\x07\x81+\n\x04\x01\x03\x0f\x05\x81N\x06\x01\x03\'\x82!\x08\x03\x01\x03\x07\x1b\x06\x01\x03\t\x84&\x06\x01\x03\x14\x87\x13\x08\x02\x01\x03\t\x827\x02\x04\x81\x12\x06\x01\x03\x0b\x81\x06\x08\x07\x01\x03\x11\x82\x1c\x06\x01\x01\x02\x81H\x06\x01\x03\x12+\x08\x06\x01\x03\x03\x84z\x06\x01\x03\x10\x836\x06\x01\x03\x08U\x06\x01\x03\x1do\x06\x01\x03\r\x811\x08\x04\x01\x03\x0e\x81\x0c\x06\x01\x03\x1a\x83x\x08\x06\x01\x03\r\x83K\x06\x01\x03\x0e\x82#\x02\x05\x81\n\x06\x01\x03\x1d\x810\x08\x04\x01\x03\r\x81=\x06\x01\x03\x10\x82\n\x06\x01\x03\x11\x81\x0e\x06\x01\x03$\x82\x1c\x06\x01\x03\x18\x81\x1c\x02\x04M\x06\x01\x03!\x83\x1e\x06\x01\x03\x12\x82A\x06\x01\x03\x18D\x06\x01\x03"\x81 \x06\x01\x03\x16\x81&\x06\x01\x03\tY\x08\x02\x01\x03\r\x84g\x06\x01\x01\x02w\x06\x01\x03\x1f\x86h\x06\x01\x03\x08r\x06\x01\x03\re\x08\x02\x01\x03\x06\x7f\x06\x01\x03\x1dl\x08\x04\x01\x03\x06\x83H\x06\x01\x03\n\x84\x01\x06\x01\x03\x0f\x848\x06\x01\x03\x105\x08\x05\x01\x03\x10\x81 \x06\x01\x03\x0e}\x08\x04\x01\x03\x0f\x82-\n\x05\x01\x03\n\x06M\x08\x03\x01\x03\r\x85M\x06\x01\x03\x0b\x81\x01\x06\x01\x03\x0f#\x08\x07\x01\x03\x1f>\x08\x05\x01\x03\x1a\x81B\x06\x01\x03\x10\x83\x05\x06\x01\x03\x0f\x81\x1c\x02\x03N\x08\x02\x01\x03\x06E\x08\x07\x01\x03\x153\x08\x02\x01\x03\x06\x84\x0c\x06\x01\x03\x1e\x84\x0f\x02\x02\x83\x0f\x06\x01\x03\x11\x8au\x08\x03\x01\x03\x06\x89x\x08\x03\x01\x03\x08\x82H\x02\x07\x814\x06\x01\x03\x19\x83&\x02\x05\x83\x06\x02\x03\x836\x02\x06\x82\x16\x08\x04\x01\x03\x0f`\x08\x04\x01\x03\nd\x08\x05\x01\x03\x10\x81\t\x08\n\x01\x03\x0e\x11\x08\x05\x01\x03#\x89E\x08\t\x01\x03\x0b\x81;\x06\x01\x03\x07O\x08\x03\x01\x03$t\x08\x04\x01\x03\x19\x82@\x02\x02\x85\x02\x06\x01\x03\x1bZ\x02\x04\x81e\x06\x01\x03\x0e\x83!\x06\x01\x01\x02=\x08\x03\x01\x03\x12\x82\x1f\x06\x01\x03\x06\x81\x07\x06\x01\x03\x12w\x08\x02\x01\x03\r\x10\x08\x03\x01\x03\x07~\x06\x01\x03\x0c\x82Q\x06\x01\x03\x0e\x82,\x06\x01\x03\t\x84 \x06\x01\x03\n\x86\x06\x08\x05\x01\x03\x11\x81\x17\x06\x01\x01\x02\x82P\x08\t\x01\x03\x12\x81D\x06\x01\x03\r\x82\x01\x08\t\x01\x03\x1e\x85\x1b\x08\t\x01\x02\x04\x87\x14\x06\x01\x03\x06\x83A\x08\x02\x01\x03\x06\x83I\x08\x06\x01\x03\x1d\x844\x06\x01\x03\x19\x82\x1b\x06\x01\x03\x13\x84\x14\x06\x01\x03\n\x82\x14\x08\x02\x01\x02\x02\x81C\x08\x06\x01\x03\x0e\x8a/\x08\x07\x01\x03\x07m\x06\x01\x03\x17\x84\x17\x06\x01\x03\x0e\x81i\x06\x01\x03\x10\x83-\x08\x05\x01\x03\x08\x1f\x06\x01\x03\t!\x08\x03\x01\x03\t\x8a6\x08\x03\x01\x03\x08H\x06\x01\x03\x0b\x86\x15\x06\x01\x03\x0bK\x06\x01\x03\x0e(\x02\x03\x82\x05\x08\x02\x01\x03\x0c\x82+\x06\x01\x03\x06]\x06\x01\x03\x12+\x02\x07\x81m\x08\x02\x01\x03\x06\x81[\x08\r\x01\x035\x81w\x08\x06\x01\x03\x10\x82c\x08\x04\x01\x03\x13\x84S\x08\x0c\x01\x02\x06\x0c\x06\x01\x03\x07|\x06\x01\x03\x18\x81/\x06\x01\x03\x08\t\x06\x01\x03\t\x816\x08\x04\x01\x03\x06\x84m\x06\x01\x01\x02\x83\x1f\x06\x01\x03\x0bJ\x06\x01\x03\x18\x82(\x02\x04I\x08\x05\x01\x03\x06\x81\\\x08\x02\x01\x03\x06\x81U\x08\x05\x01\x03\x07\r\x02\x02\x83J\x06\x01\x03\x08-\x02\x03\x84P\x06\x01\x03\n\x82\x1b\x08\x04\x01\x03\r\x87Y\x08\t\x01\x03\x19j\x08\x08\x01\x03\x11\x812\x08\x07\x01\x03\x16\x81K\x08\x02\x01\x03\x06Q\x02\x05\x839\x02\x03\x81B\x06\x01\x03\x0b\x08\x01s\x85e\x06\x01\x01\x02\xa0A\x08\x0b\x01\x02\x02\x920\x08\x02\x01\x03\x05\xea:\x02\x06\xd3E\x08\x03\x01\x03\x04\x9a*\x06\x01\x03\n\x82x\x02\x04\x82K\x02\x05\x98a\x06\x01\x03\x10\x84v\x02\x02\xac4\x06\x01\x03\x18\x82.\x08\x08\x01\x03\x1c\x8dJ\x08\x02\x01\x02\x02\x94$\x06\x01\x03\x0e\x06\x03ing\x82\x06\x08\x02\x01\x02\x02\x8c\x19\x06\x01\x03\x16\x88x\x08\x02\x01\x03\x0b\x87\x06\x06\x01\x01\x02$\x02\x08^\x06\x01\x03\t\x84&\x06\x01\x01\x02\x84j\x06\x01\x01\x02r\x06\x01\x01\x02\x82@\x06\x01\x01\x02\x81K\x0e\x03\x01\x02\x02\x01\x03\x05\x98U\x08\x04\x01\x03\t\x86e\x06\x01\x01\x02\x82\x04\x06\x01\x03\x10\x01\x06\x01\x01\x02\x91#\x08\n\x01\x02\x04\x87\x0c\x06\x01\x03\x06\x82Z\x06\x01\x03\x0b\x85+\x08\x05\x01\x02\x05\x91A\x08\x04\x01\x03\x07\x9au\x08\x03\x01\x02\x03\x91N\x06\x01\x03\x0f\x87)\x0e\x02\x01\x02\x02\x01\x03\x05\x8e*\x08\x04\x01\x03\x02\x8f\x00\x08\x02\x01\x02\x02h\x02\x06\x811\x06\x01\x01\x02\x885\x02\x02\x86K\x08\x03\x01\x03\x08\x88H\x06\x01\x03\x0b\x81n\x06\x01\x01\x02\x82X\x08\x04\x01\x03\x05\x89A\x02\t\x8d2\x06\x01\x01\x02\x81`\x08\x02\x01\x02\x02\x8cs\x08\x0b\x01\x02\x04\x85\x1d\x08\x03\x01\x03\x0b\x89A\x06\x01\x01\x02\x8fo\x08\t\x01\x02\x03\x8b\x1a\x08\x02\x01\x03\x03\x84s\x06\x01\x01\x02\x83s\x08\x08\x01\x02\x04\x81y\x06\x01\x03\x19\x8a<\x06\x01\x03\x0b}\x08\x08\x01\x02\x02\x9cm\x08\x01\x03\x12\x19\x82a\x08\x04\x81l\r\x88(T'}
```

---

## Table: `clues_fts_docsize`

**Row count:** 327,189

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `sz` | BLOB | - |

### Sample Data

```
{'id': 1, 'sz': b'\x06\x01\x02\n'}
{'id': 2, 'sz': b'\x07\x01\x01\x0e'}
{'id': 3, 'sz': b'\t\x01\x01\x10'}
```

---

## Table: `clues_fts_idx`

**Row count:** 6,263

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `segid` | ANY | PRIMARY KEY, NOT NULL |
| `term` | ANY | PRIMARY KEY, NOT NULL |
| `pgno` | ANY | - |

### Indexes

- `sqlite_autoindex_clues_fts_idx_1`: UNIQUE (segid, term)

### Sample Data

```
{'segid': 1, 'term': b'', 'pgno': 2}
{'segid': 1, 'term': b'0100', 'pgno': 6}
{'segid': 1, 'term': b'0110', 'pgno': 8}
```

---

## Table: `definition_answers`

**Row count:** 162,185

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `definition` | TEXT | NOT NULL |
| `answer` | TEXT | NOT NULL |
| `source` | TEXT | - |
| `frequency` | INTEGER | DEFAULT 1 |

### Indexes

- `idx_da_answer`: (answer)
- `idx_da_definition`: (definition)
- `sqlite_autoindex_definition_answers_1`: UNIQUE (definition, answer)

### Sample Data

```
{'id': 1, 'definition': 'double-definition', 'answer': 'MOPED', 'source': 'telegraph', 'frequency': 2}
{'id': 2, 'definition': 'decorated', 'answer': 'ELABORATE', 'source': 'telegraph', 'frequency': 1}
{'id': 3, 'definition': 'met', 'answer': 'SATISFIED', 'source': 'telegraph', 'frequency': 3}
```

---

## Table: `definition_mappings`

**Row count:** 57,289

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `phrase` | TEXT | NOT NULL |
| `target` | TEXT | NOT NULL |

### Indexes

- `idx_definition_mappings_unique`: UNIQUE (phrase, target)

### Sample Data

```
{'id': 1, 'phrase': 'a', 'target': 'alpha'}
{'id': 2, 'phrase': 'a', 'target': 'an'}
{'id': 3, 'phrase': 'a', 'target': 'ar'}
```

---

## Table: `indicator_type_map`

**Row count:** 0

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `raw_type` | TEXT | PRIMARY KEY |
| `unified_type` | TEXT | NOT NULL |

### Indexes

- `sqlite_autoindex_indicator_type_map_1`: UNIQUE (raw_type)


---

## Table: `indicators`

**Row count:** 18,496

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `word` | TEXT | NOT NULL |
| `wordplay_type` | TEXT | NOT NULL |
| `subtype` | TEXT | - |
| `confidence` | TEXT | DEFAULT 'high' |

### Indexes

- `sqlite_autoindex_indicators_1`: UNIQUE (word, wordplay_type, subtype)

### Sample Data

```
{'id': 1, 'word': 'abandon', 'wordplay_type': 'anagram', 'subtype': None, 'confidence': 'high'}
{'id': 2, 'word': 'abandoned', 'wordplay_type': 'anagram', 'subtype': None, 'confidence': 'high'}
{'id': 3, 'word': 'aberrant', 'wordplay_type': 'anagram', 'subtype': None, 'confidence': 'high'}
```

---

## Table: `sqlite_sequence`

**Row count:** 6

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `name` | ANY | - |
| `seq` | ANY | - |

### Sample Data

```
{'name': 'clues', 'seq': 327189}
{'name': 'indicators', 'seq': 18497}
{'name': 'wordplay', 'seq': 24666}
```

---

## Table: `substitutions`

**Row count:** 7

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `original_word` | TEXT | NOT NULL |
| `substitution` | TEXT | NOT NULL |
| `context` | TEXT | - |
| `confidence` | REAL | DEFAULT 0.9 |
| `discovered_date` | TEXT | DEFAULT CURRENT_TIMESTAMP |

### Indexes

- `sqlite_autoindex_substitutions_1`: UNIQUE (original_word, substitution)

### Sample Data

```
{'id': 1, 'original_word': 'odd', 'substitution': 'RUM', 'context': 'synonym for strange', 'confidence': 0.9, 'discovered_date': '2025-12-07 05:05:00'}
{'id': 2, 'original_word': 'airline', 'substitution': 'BA', 'context': 'British Airways', 'confidence': 0.9, 'discovered_date': '2025-12-07 05:05:00'}
{'id': 3, 'original_word': 'german', 'substitution': 'UND', 'context': 'German language', 'confidence': 0.9, 'discovered_date': '2025-12-07 05:05:00'}
```

---

## Table: `successful_patterns`

**Row count:** 5

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `clue_text` | TEXT | NOT NULL |
| `answer` | TEXT | NOT NULL |
| `mechanism` | TEXT | NOT NULL |
| `explanation` | TEXT | - |
| `complexity` | TEXT | DEFAULT 'intermediate' |
| `discovered_date` | TEXT | DEFAULT CURRENT_TIMESTAMP |
| `success_count` | INTEGER | DEFAULT 1 |

### Indexes

- `sqlite_autoindex_successful_patterns_1`: UNIQUE (clue_text, answer)

### Sample Data

```
{'id': 1, 'clue_text': "Odd airline's dance (5)", 'answer': 'RUMBA', 'mechanism': 'charade', 'explanation': 'Odd = RUM, airline = BA, RUM + BA = RUMBA', 'complexity': 'intermediate', 'discovered_date': '2025-12-07 05:05:00', 'success_count': 1}
{'id': 2, 'clue_text': 'Suffer in German and therefore in Latin (7)', 'answer': 'UNDERGO', 'mechanism': 'charade', 'explanation': 'German = UND, Latin = ERGO, UND + ERGO = UNDERGO', 'complexity': 'intermediate', 'discovered_date': '2025-12-07 05:05:00', 'success_count': 1}
{'id': 3, 'clue_text': 'Cheats with every second bullet boring outraged readers (10)', 'answer': 'ADULTERERS', 'mechanism': 'multi-mechanism', 'explanation': 'Every second bullet = ULT, boring = insert, outraged readers = anagram of READERS', 'complexity': 'intermediate', 'discovered_date': '2025-12-07 05:05:00', 'success_count': 1}
```

---

## Table: `synonyms`

**Row count:** 78,669

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `word` | TEXT | PRIMARY KEY |
| `synonyms` | TEXT | - |
| `fetched_at` | TIMESTAMP | DEFAULT NULL |
| `source` | TEXT | DEFAULT 'merriam-webster' |

### Indexes

- `idx_synonyms_word`: (word)
- `sqlite_autoindex_synonyms_1`: UNIQUE (word)

### Sample Data

```
{'word': 'ability', 'synonyms': '["competency", "capableness", "faculty", "capability", "competence", "capacity", "talent", "might", "skill", "streamed", "we can do with it (7)", "kindled into"]', 'fetched_at': '2025-12-03T09:32:31.408336', 'source': 'merriam-webster-thesaurus'}
{'word': 'able', 'synonyms': '["good", "fit", "expert", "chart minus", "well", "suitable", "showing talent", "clever", "bright", "chipper", "talented", "wholesome", "hearty", "sound", "skilful", "robust", "qualified", "strong", "healthy", "sharp", "adroit", "efficient", "gifted", "capable", "bouncing", "having power", "skilled", "competent", "whole", "equal", "hale"]', 'fetched_at': '2025-12-03T09:32:32.016773', 'source': 'merriam-webster-intermediate'}
{'word': 'about', 'synonyms': '["veering", "astir", "repealing", "strays", "bent", "on", "twist", "curl", "next to", "curves", "backtracking", "by", "rounding", "strikes down", "sheer", "bows", "countermand", "as regards", "yawed", "fairly", "curve", "as far as", "arced", "bowed", "crooked", "straying", "countermanding", "reversal", "arcs", "skewed", "revoking", "reverses", "circles", "back", "arc", "much", "spirals", "volte-face", "turns", "sheering", "yaw", "sleepless", "arches", "spiral", "curved", "wound", "nigh", "deviates", "reawakened", "rounded", "cutting", "hooking", "turnabout", "insomniac", "all but", "zagging", "countermands", "almost", "about-turns", "overturn", "apropos of", "annulled", "dealing with", "zags", "revived", "cuts", "zigzagging", "revoked", "weave", "wandered", "conscious", "bow", "deviating", "oberon", "turning", "twisting", "skews", "breaking", "strayed", "treasure", "backtracks", "wavering", "backtrack", "hooks", "wavered", "fair", "struck down", "roughly", "round", "circled", "turnarounds", "weaved", "turn", "borderline", "curling", "U-turns", "loop", "switch", "revert", "here and there", "pivoted", "stray", "toward", "turnabouts", "coil", "sweeping", "abrogate", "coils", "twisted", "zigging", "respecting", "hooked", "sweep", "practically", "annulling", "swerve", "wandering", "loops", "slew", "weaves", "zag", "awake", "cut", "crook", "somewhere", "zigs", "sweeps", "abrogated", "hook", "flip-flop", "turnaround", "sheered", "around", "broke", "overturning", "zigged", "winds", "circle", "revoke", "wind", "bend", "more or less", "breaks", "eager", "wakened", "rescinding", "abrogates", "sleeper", "swerved", "overturned", "annul", "awakened", "curls", "roused", "curled", "feckly", "pivot", "rousted", "spectre", "zagged", "aroused", "revokes", "bends", "looped", "reversed", "wavers", "looping", "swerving", "repealed", "veer", "virtually", "apropos", "fight", "arching", "spiraled", "countermanded", "plus or minus", "wheeling", "as for", "deviated", "most", "yaws", "like", "waver", "break", "rescind", "swerves", "wide-awake", "overturns", "nearly", "of", "zigzagged", "U-turn", "switched", "weaving", "deviate", "zigzag", "up", "backtracked", "gamekeeper", "across", "bowing", "switching", "backward", "wheeled", "pivots", "arcing", "circa", "swept", "skewing", "volte-faces", "switches", "flip-flops", "reversing", "near", "spiraling", "some", "through", "say", "sheers", "repeals", "zig", "regarding", "skew", "rescinds", "approximately", "crooking", "concerning", "not far away", "reverted", "striking down", "bending", "zigzags", "wheels", "veers", "over", "wheel", "touching", "wanders", "circling", "curving", "rescinded", "arch", "twists", "repeal", "giveortake", "arched", "yawing", "turned", "rounds", "as to", "about-turn", "slews", "annuls", "classes", "coiling", "reverts", "reversals", "abrogating", "passim", "crooks", "pivoting", "reverse", "well-nigh", "throughout", "wander", "aware", "about-face", "winding", "wakeful", "reverting", "strike down", "coiled", "veered"]', 'fetched_at': '2025-12-03T09:32:32.589511', 'source': 'merriam-webster-thesaurus'}
```

---

## Table: `synonyms_pairs`

**Row count:** 606,400

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `word` | TEXT | NOT NULL |
| `synonym` | TEXT | NOT NULL |

### Indexes

- `idx_synonyms_pairs_unique`: UNIQUE (word, synonym)

### Sample Data

```
{'id': 3012971, 'word': 'continent', 'synonym': '"happy around"'}
{'id': 3012972, 'word': 'compromise', 'synonym': '"concession"'}
{'id': 3012973, 'word': '"chance"', 'synonym': 'are'}
```

---

## Table: `wordplay`

**Row count:** 771

### Columns

| Column | Type | Constraints |
|--------|------|-------------|
| `id` | INTEGER | PRIMARY KEY |
| `indicator` | TEXT | NOT NULL |
| `substitution` | TEXT | NOT NULL |
| `category` | TEXT | - |
| `frequency` | INTEGER | DEFAULT 0 |
| `confidence` | TEXT | DEFAULT 'high' |
| `notes` | TEXT | - |

### Indexes

- `sqlite_autoindex_wordplay_1`: UNIQUE (indicator, substitution)

### Sample Data

```
{'id': 1, 'indicator': 'north', 'substitution': 'N', 'category': 'single_letter', 'frequency': 0, 'confidence': 'high', 'notes': ''}
{'id': 2, 'indicator': 'northern', 'substitution': 'N', 'category': 'single_letter', 'frequency': 0, 'confidence': 'high', 'notes': ''}
{'id': 3, 'indicator': 'south', 'substitution': 'S', 'category': 'single_letter', 'frequency': 0, 'confidence': 'high', 'notes': ''}
```

---

## Triggers

### `clues_ad`

```sql
CREATE TRIGGER clues_ad AFTER DELETE ON clues BEGIN
        INSERT INTO clues_fts(clues_fts, rowid, clue_text, answer, definition, explanation)
        VALUES('delete', old.id, old.clue_text, old.answer, old.definition, old.explanation);
    END
```

### `clues_ai`

```sql
CREATE TRIGGER clues_ai AFTER INSERT ON clues BEGIN
        INSERT INTO clues_fts(rowid, clue_text, answer, definition, explanation)
        VALUES (new.id, new.clue_text, new.answer, new.definition, new.explanation);
    END
```
