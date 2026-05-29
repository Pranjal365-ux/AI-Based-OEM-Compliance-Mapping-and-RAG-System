import chromadb
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

client = chromadb.PersistentClient(path="./VectorDB")
collection = client.get_collection("oem_knowledge_base")

# Find full tables
results = collection.get(where={"chunk_type": "table_full"}, include=["documents"])

from check_tables import parse_markdown_table

for i, doc in enumerate(results["documents"]):
    headers, rows = parse_markdown_table(doc)
    # Check if this table has FG-7081 or FG-7121 specs
    has_specs = any("FG-7" in h for h in headers)
    if has_specs:
        print(f"\n==================== TABLE {i+1} ====================")
        print("Headers:", headers)
        print("All rows:")
        for r_idx, r in enumerate(rows):
            print(f"  Row {r_idx:02d}: {r}")
