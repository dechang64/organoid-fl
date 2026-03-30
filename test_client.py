"""Python client for Organoid VectorDB gRPC service"""
import sys, os, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
import grpc
import vectordb_pb2 as pb
import vectordb_pb2_grpc as pb_grpc

def main():
    channel = grpc.insecure_channel('localhost:50058')
    grpc.channel_ready_future(channel).result(timeout=10)
    stub = pb_grpc.VectorDBStub(channel)

    # 1. Insert 100 random vectors (simulating organoid image features)
    print("=== Inserting 100 vectors (dim=128) ===")
    vectors = []
    for i in range(100):
        vectors.append(pb.Vector(
            id=f"organoid_{i:04d}",
            values=[random.gauss(0, 1) for _ in range(128)],
            metadata={"source": f"client_{i % 3}", "type": "organoid"}
        ))

    start = time.time()
    resp = stub.Insert(pb.InsertRequest(vectors=vectors))
    elapsed = time.time() - start
    print(f"  Inserted: {resp.inserted} vectors in {elapsed*1000:.1f}ms")

    # 2. Check stats
    stats = stub.Stats(pb.StatsRequest())
    print(f"\n=== DB Stats ===")
    print(f"  Total vectors: {stats.total_vectors}")
    print(f"  Dimension: {stats.dimension}")

    # 3. Search for nearest neighbors
    print(f"\n=== Searching (k=5) ===")
    query = [random.gauss(0, 1) for _ in range(128)]
    start = time.time()
    results = stub.Search(pb.SearchRequest(query=query, k=5))
    elapsed = time.time() - start
    print(f"  Search time: {elapsed*1000:.1f}ms")
    for r in results.results:
        print(f"    {r.id}: distance={r.distance:.4f}, metadata={dict(r.metadata)}")

    # 4. Delete some vectors
    print(f"\n=== Deleting 3 vectors ===")
    del_resp = stub.Delete(pb.DeleteRequest(ids=["organoid_0000", "organoid_0001", "organoid_0002"]))
    print(f"  Deleted: {del_resp.deleted}")

    # 5. Check stats again
    stats = stub.Stats(pb.StatsRequest())
    print(f"  Total vectors after delete: {stats.total_vectors}")

    print("\n=== All tests passed! ===")
    channel.close()

if __name__ == '__main__':
    main()
