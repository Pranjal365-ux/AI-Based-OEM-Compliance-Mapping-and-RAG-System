from rfp.rfp_extractor import RFPRequirementExtractor

extractor = RFPRequirementExtractor()

# Step 1: Load and classify chunks
pages = extractor.extract_pages("C:/Users/Pranjal/OneDrive/Desktop/Starlight/RFPs/datacenter_2023-12-29-16-00-03_56abbc0d86ad9ccc650d442bbabc286a.pdf")
store = extractor.chunk_and_classify(pages)

categories = store.categories()
print("\nProducts / Categories Found:\n")

for i, category in enumerate(categories):
    chunks = store.chunks_for(category)
    print(f"{i}: {category} ({len(chunks)} chunk(s))")

# Step 2: Extract requirements for the first category found
if categories:
    selection = categories[2]
    print(f"\nExtracting requirements for category: {selection}")
    requirements = extractor.extract_for_category(store, selection)
    print(f"\nExtracted {len(requirements)} requirements")
else:
    print("\nNo categories matched.")
