from rfp.rfp_extractor import RFPRequirementExtractor

extractor = RFPRequirementExtractor()

pages = extractor.extract_pages("C:/Users/Pranjal/OneDrive/Desktop/Starlight/RFPs/datacenter_2023-12-29-16-00-03_56abbc0d86ad9ccc650d442bbabc286a.pdf")

manifest = extractor.discover_products(pages)

print("\nProducts Found:\n")

for i, product in enumerate(manifest.products):
    print(
        f"{i}: {product.product} "
        f"(Pages {product.start_page}-{product.end_page})"
    )

selection = 0

requirements = extractor.extract_for_product(
    pages,
    manifest.products[selection]
)

print(f"\nExtracted {len(requirements)} requirements")