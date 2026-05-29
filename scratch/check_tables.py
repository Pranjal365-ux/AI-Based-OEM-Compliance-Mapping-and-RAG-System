import re
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

def parse_markdown_table(table_text: str) -> tuple[list[str], list[list[str]]]:
    table_text = table_text.replace("[TABLE]", "").replace("[/TABLE]", "").strip()
    lines = [l.strip() for l in table_text.split("\n") if l.strip() and "|" in l]
    if len(lines) < 2:
        return [], []
        
    def split_row(row_str: str) -> list[str]:
        s = row_str.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [cell.strip() for cell in s.split("|")]

    headers = split_row(lines[0])
    
    data_start_idx = 1
    if len(lines) > 1:
        second_row_cells = split_row(lines[1])
        is_sep = all(re.match(r"^[\s\-\:\+]*$", c) for c in second_row_cells)
        if is_sep:
            data_start_idx = 2
            
    data_rows = []
    for line in lines[data_start_idx:]:
        cells = split_row(line)
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        data_rows.append(cells)
        
    return headers, data_rows

# Fetch some table texts from the database!
import chromadb
client = chromadb.PersistentClient(path="./VectorDB")
collection = client.get_collection("oem_knowledge_base")

# Find full tables
results = collection.get(where={"chunk_type": "table_full"}, include=["documents"])
for i, doc in enumerate(results["documents"]):
    print(f"\n==================== TABLE {i+1} ====================")
    print("Raw doc snippet:")
    print(doc[:300])
    headers, rows = parse_markdown_table(doc)
    print("\nParsed Headers:", headers)
    print(f"Parsed {len(rows)} rows. First 2 rows:")
    for r in rows[:2]:
        print("  ", r)
