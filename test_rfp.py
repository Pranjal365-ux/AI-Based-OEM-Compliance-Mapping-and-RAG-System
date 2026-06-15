from rfp.rfp_extractor import RFPRequirementExtractor

extractor = RFPRequirementExtractor()

PDF_PATH = "C:/Users/Pranjal/OneDrive/Desktop/Starlight/RFPs/datacenter_2023-12-29-16-00-03_56abbc0d86ad9ccc650d442bbabc286a.pdf"

# Step 1: Load the PDF and report how many pages it has
pages = extractor.extract_pages(PDF_PATH)
print(f"\nDocument has {len(pages)} page(s).")

# Step 2: User selects the page range to scan (e.g. the section of the RFP
# that covers the product they're checking compliance for)
start_page, end_page = 1, 5

# Step 3: Extract every requirement found on those pages — no product /
# category classification is performed
requirements = extractor.extract_requirements_from_range(pages, start_page, end_page)

print(f"\nExtracted {len(requirements)} requirement(s) from pages {start_page}-{end_page}:\n")
for req in requirements:
    flag = "MUST" if req.mandatory else "should"
    value = f" {req.operator} {req.value} {req.unit or ''}".rstrip() if req.unit else ""
    print(f"  [{req.requirement_id}] ({req.category}) {flag}: {req.requirement}{value}")
    print(f"      source: {req.source_text!r}  ({req.section})")

# Step 4 (optional): write everything to JSON and embed into Chroma for the
# next stage (matching requirements against the OEM knowledge base and
# generating a compliance report for the best-fitting products).
#
# result = extractor.run(PDF_PATH, start_page, end_page)