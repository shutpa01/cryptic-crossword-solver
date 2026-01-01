import random


# ----------------------------
# Public API
# ----------------------------

def get_cohort(conn, criteria):
    """
    Return a list of clue rows according to selection criteria.

    The orchestrator should treat this as a black box.
    """
    source = criteria.get("source", "clues")
    limit = criteria.get("limit")
    where = criteria.get("where", {})
    order = criteria.get("order", "random")
    seed = criteria.get("seed")

    if seed is not None:
        random.seed(seed)

    sql, params = _build_query(source, where, order, limit)

    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()

    # Optional post-filter (rare, but useful)
    post_filter = criteria.get("post_filter")
    if post_filter:
        rows = [r for r in rows if post_filter(r)]

    return rows


# ----------------------------
# Internal helpers
# ----------------------------

def _build_query(source, where, order, limit):
    """
    Build SQL safely from declarative criteria.
    """
    clauses = []
    params = []

    for col, val in where.items():
        if isinstance(val, (list, tuple, set)):
            placeholders = ",".join("?" for _ in val)
            clauses.append(f"{col} IN ({placeholders})")
            params.extend(val)
        else:
            clauses.append(f"{col} = ?")
            params.append(val)

    where_sql = ""
    if clauses:
        where_sql = "WHERE " + " AND ".join(clauses)

    if order == "random":
        order_sql = "ORDER BY RANDOM()"
    elif order == "id":
        order_sql = "ORDER BY rowid"
    else:
        raise ValueError(f"Unsupported order: {order}")

    limit_sql = ""
    if limit:
        limit_sql = f"LIMIT {int(limit)}"

    sql = f"""
        SELECT clue_text, enumeration, answer
        FROM {source}
        {where_sql}
        {order_sql}
        {limit_sql}
    """

    return sql, params
