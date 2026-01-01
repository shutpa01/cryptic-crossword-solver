# solver/selection/presets.py

# Cohort-selection presets.
# These contain NO solver logic — only declarative selection criteria.


CRITERIA_RANDOM = {
    "source": "clues",
    "limit": 1000,
    "order": "random",
}

CRITERIA_ALL = {
    "source": "clues",
    "order": "id",
}

CRITERIA_DD_ONLY = {
    "source": "clues",
    "where": {
        "wordplay_type": "double_definition",
    },
    "limit": 10,   # explicit: no cap
    "order": "id",   # deterministic ordering
}

CRITERIA_HIDDEN_ONLY = {
    "source": "clues",
    "where": {
        "wordplay_type": "hidden",
    },
}

# Single fixed selector symbol (master_solver imports THIS only)
CURRENT_CRITERIA = CRITERIA_RANDOM

# Shared paths — these are declared here for single-source configuration
COHORT_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cohort.txt"

DB_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
WORDLIST_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\wordlist.txt"
