# import chromadb

# client = chromadb.PersistentClient(path="./VectorDB")
# collection = client.get_collection("oem_knowledge_base")

# results = collection.query(
#     query_texts=["SSL inspection throughput"],
#     n_results=3
# )

# for i in range(len(results["ids"][0])):
#     print(f"\n==================== RESULT {i+1} ====================")
#     print("ID:", results["ids"][0][i])
#     print("METADATA:", results["metadatas"][0][i])
#     print("TEXT:")
#     print(repr(results["documents"][0][i]))



import chromadb

client = chromadb.PersistentClient(path="./VectorDB")
collection = client.get_collection("oem_knowledge_base")

results = collection.get(
where={"chunk_type":"spec_row"},
include=["documents","metadatas"]
)

print("\nTotal spec rows:",len(results["documents"]))

for i in range(min(10,len(results["documents"]))):

    print("\n"+"="*60)

    print(results["metadatas"][i])

    print(results["documents"][i][:500])

