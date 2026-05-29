from sentence_transformers import SentenceTransformer
import chromadb
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./VectorDB")

collection = client.get_collection("oem_knowledge_base")

print("="*70)
print("DATABASE STATS")
print("="*70)
total_vectors = collection.count()
print("Total Vectors:", total_vectors)

# Fetch all metadata to compute chunk types
all_data = collection.get(include=["metadatas", "documents"])
metadatas = all_data["metadatas"]
documents = all_data["documents"]

type_counts = {}
sample_specs = []

for idx, meta in enumerate(metadatas):
    ctype = meta.get("chunk_type", "un-tagged")
    type_counts[ctype] = type_counts.get(ctype, 0) + 1
    if ctype == "spec_row":
        sample_specs.append((meta, documents[idx]))

print("\nChunk Counts by Type:")
for ctype, count in sorted(type_counts.items()):
    print(f"  {ctype:<15}: {count} chunks")

if sample_specs:
    print("\n" + "="*70)
    print("SAMPLE GENERATED SPEC CHUNKS:")
    print("="*70)
    for idx, (meta, doc) in enumerate(sample_specs[:3], 1):
        print(f"\nSample {idx}:")
        print(f"Metadata: {meta}")
        print(f"Content:\n{doc}")
        print("-" * 55)

print("\n" + "="*70)
print("RUNNING RETRIEVAL QUERIES")
print("="*70)

queries = [
    "Supports TLS 1.3 inspection",
    "Threat protection throughput above 500 Gbps",
    "Zero Trust Network Access support",
    "SSL inspection throughput"
]

for q in queries:
    print("\n" + "="*70)
    print("QUERY:", q)

    embedding = model.encode(q).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    docs = results["documents"][0]

    for i, doc in enumerate(docs,1):
        print(f"\nResult {i}")
        print("-"*50)
        print(doc[:800])