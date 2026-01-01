"""
Extract complete schema from cryptic.db SQLite database.
Run: python extract_schema.py
Output: schema_documentation.md in the same directory
"""
import sqlite3
import os
from datetime import datetime

def extract_schema():
    """Extract complete schema from SQLite database."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic.db"
    
    print(f"Connecting to: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    output = []
    output.append("# Database Schema: cryptic.db")
    output.append("")
    output.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("")
    output.append("---")
    output.append("")
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    output.append("## Tables Overview")
    output.append("")
    output.append(f"Total tables: {len(tables)}")
    output.append("")
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) FROM [{t}]")
        count = cursor.fetchone()[0]
        output.append(f"- **{t}**: {count:,} rows")
    output.append("")
    output.append("---")
    output.append("")
    
    # For each table, get detailed info
    for table in tables:
        output.append(f"## Table: `{table}`")
        output.append("")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
        count = cursor.fetchone()[0]
        output.append(f"**Row count:** {count:,}")
        output.append("")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info([{table}])")
        columns = cursor.fetchall()
        output.append("### Columns")
        output.append("")
        output.append("| Column | Type | Constraints |")
        output.append("|--------|------|-------------|")
        for col in columns:
            cid, name, col_type, notnull, default, pk = col
            constraints = []
            if pk:
                constraints.append("PRIMARY KEY")
            if notnull:
                constraints.append("NOT NULL")
            if default is not None:
                constraints.append(f"DEFAULT {default}")
            constraint_str = ", ".join(constraints) if constraints else "-"
            output.append(f"| `{name}` | {col_type or 'ANY'} | {constraint_str} |")
        output.append("")
        
        # Get indexes
        cursor.execute(f"PRAGMA index_list([{table}])")
        indexes = cursor.fetchall()
        if indexes:
            output.append("### Indexes")
            output.append("")
            for idx in indexes:
                idx_name = idx[1]
                unique = "UNIQUE " if idx[2] else ""
                cursor.execute(f"PRAGMA index_info([{idx_name}])")
                idx_cols = [row[2] for row in cursor.fetchall()]
                output.append(f"- `{idx_name}`: {unique}({', '.join(idx_cols)})")
            output.append("")
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list([{table}])")
        fks = cursor.fetchall()
        if fks:
            output.append("### Foreign Keys")
            output.append("")
            for fk in fks:
                output.append(f"- `{fk[3]}` → `{fk[2]}.{fk[4]}`")
            output.append("")
        
        # Show sample data (first 3 rows)
        cursor.execute(f"SELECT * FROM [{table}] LIMIT 3")
        samples = cursor.fetchall()
        if samples:
            col_names = [desc[0] for desc in cursor.description]
            output.append("### Sample Data")
            output.append("")
            output.append("```")
            for row in samples:
                output.append(str(dict(zip(col_names, row))))
            output.append("```")
        output.append("")
        output.append("---")
        output.append("")
    
    # Get views
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='view' ORDER BY name")
    views = cursor.fetchall()
    if views:
        output.append("## Views")
        output.append("")
        for name, sql in views:
            output.append(f"### `{name}`")
            output.append("")
            output.append("```sql")
            output.append(sql if sql else "(no SQL available)")
            output.append("```")
            output.append("")
    
    # Get triggers
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name")
    triggers = cursor.fetchall()
    if triggers:
        output.append("## Triggers")
        output.append("")
        for name, sql in triggers:
            output.append(f"### `{name}`")
            output.append("")
            output.append("```sql")
            output.append(sql if sql else "(no SQL available)")
            output.append("```")
            output.append("")
    
    conn.close()
    
    # Write output
    output_path = os.path.join(script_dir, 'schema_documentation.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output))
    
    print(f"Schema documentation written to: {output_path}")
    print("\n" + "="*50)
    print("SCHEMA PREVIEW:")
    print("="*50)
    print("\n".join(output[:50]))  # Print first 50 lines
    return "\n".join(output)

if __name__ == "__main__":
    extract_schema()
