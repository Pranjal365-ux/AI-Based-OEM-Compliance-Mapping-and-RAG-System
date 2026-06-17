"""
Quick test / demo of the RFP extraction pipeline.

Usage:
  python test_rfp.py

What it does:
  1. Extracts all pages from the PDF
  2. Extracts every requirement on pages START_PAGE–END_PAGE
  3. Writes  data/requirements/<stem>_pp{start}-{end}.json
  4. Embeds  into data/vector_store/ via bge-m3 (same Chroma as OEM KB)
  5. Prints a summary to the terminal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rfp.rfp_extractor import RFPRequirementExtractor

# ── Configure these two things ────────────────────────────────────────────────
PDF_PATH   = "C:/Users/Pranjal/OneDrive/Desktop/Starlight/RFPs/datacenter_2023-12-29-16-00-03_56abbc0d86ad9ccc650d442bbabc286a.pdf"
START_PAGE = 3
END_PAGE   = 5
# ─────────────────────────────────────────────────────────────────────────────

extractor = RFPRequirementExtractor()

# Quick peek at document length before running
pages = extractor.extract_pages(PDF_PATH)
print(f"\nDocument: {len(pages)} page(s) total.")
print(f"Scanning pages {START_PAGE}–{END_PAGE} …\n")

# Full pipeline: extract → save JSON → embed into Chroma
result = extractor.run(PDF_PATH, START_PAGE, END_PAGE, embed=True)

# Summary
print(f"\n{'─'*65}")
print(f"Requirements found : {result['requirement_count']}")
print(f"JSON saved to      : {result['json_path']}")
print(f"Chroma collection  : {result['chroma_collection']}")
print(f"{'─'*65}\n")

# Print each requirement
for req in result["requirements"]:
    flag  = "MUST" if req["mandatory"] else "should"
    value = ""
    if req.get("unit"):
        value = f"  [{req['operator']} {req['value']} {req['unit']}]"
    elif req.get("value") and req["value"] != "true":
        value = f"  [{req['operator']} {req['value']}]"
    print(f"  [{req['requirement_id']}] ({req['category']}) {flag}: {req['requirement']}{value}")
    print(f"      ↳ {req['source_text'][:120]!r}  ({req['section']})")