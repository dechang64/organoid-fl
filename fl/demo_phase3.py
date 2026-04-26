#!/usr/bin/env python3
"""
Phase 3 Demo: HNSW Index + Blockchain Audit Log + Web Dashboard

Demonstrates the complete Organoid VectorDB platform with:
1. HNSW approximate nearest neighbor search (replacing brute-force)
2. Blockchain-style immutable audit logging
3. Web dashboard for monitoring

Usage:
    # Terminal 1: Start the server
    python3 demo_phase3.py --server

    # Terminal 2: Run the demo
    python3 demo_phase3.py --demo

    # Terminal 3: Open dashboard
    # Navigate to http://localhost:8080 in your browser
"""

import argparse
import grpc
import numpy as np
import time
import subprocess
import sys
import os
import json
import threading

# Add proto generated path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'proto'))
import vector_db_pb2 as pb2
import vector_db_pb2_grpc as pb2_grpc


def generate_organoid_features(n_samples=500, dim=128):
    """Generate synthetic organoid image features."""
    np.random.seed(42)
    # Simulate different organoid types with cluster centers
    n_types = 5
    centers = np.random.randn(n_types, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    features = []
    labels = []
    for i in range(n_samples):
        center = centers[i % n_types]
        noise = np.random.randn(dim).astype(np.float32) * 0.1
        feature = center + noise
        feature /= np.linalg.norm(feature)
        features.append(feature)
        labels.append(f"type_{i % n_types}")

    return features, labels


def benchmark_hnsw(stub, features, labels):
    """Benchmark HNSW search performance."""
    print("\n" + "=" * 60)
    print("📊 HNSW Search Benchmark")
    print("=" * 60)

    # Insert vectors
    print(f"\n📥 Inserting {len(features)} vectors...")
    start = time.time()
    batch_size = 50
    for i in range(0, len(features), batch_size):
        batch = features[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        vectors = [
            pb2.Vector(
                id=f"organoid_{j:04d}",
                values=batch[j - i].tolist(),
                metadata={"type": batch_labels[j - i], "source": "synthetic"}
            )
            for j in range(i, i + len(batch))
        ]
        response = stub.Insert(pb2.InsertRequest(vectors=vectors))
    insert_time = time.time() - start
    print(f"  ✅ Inserted {len(features)} vectors in {insert_time:.2f}s ({len(features)/insert_time:.0f} vec/s)")

    # Get stats
    stats = stub.Stats(pb2.StatsRequest())
    print(f"\n📈 Database Stats:")
    print(f"  Total vectors: {stats.total_vectors}")
    print(f"  Dimension: {stats.dimension}")
    print(f"  Index size: {stats.index_size}")

    # Benchmark search with different k values
    print(f"\n🔍 Search Benchmark:")
    query = features[0].tolist()

    for k in [1, 5, 10, 20, 50]:
        times = []
        for _ in range(10):
            start = time.time()
            response = stub.Search(pb2.SearchRequest(
                query=query,
                k=k,
            ))
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        p99_time = np.percentile(times, 99) * 1000
        print(f"  k={k:3d}: avg={avg_time:.2f}ms, p99={p99_time:.2f}ms, results={len(response.results)}")

    # Verify search quality
    print(f"\n✅ Search Quality Check:")
    response = stub.Search(pb2.SearchRequest(query=query, k=5))
    for r in response.results:
        print(f"  {r.id}: distance={r.distance:.4f}, type={r.metadata.get('type', 'N/A')}")

    return stats


def demonstrate_audit_log(stub):
    """Demonstrate blockchain audit log via the web API."""
    import urllib.request

    print("\n" + "=" * 60)
    print("⛓️  Blockchain Audit Log")
    print("=" * 60)

    try:
        # Fetch audit log from HTTP API
        req = urllib.request.Request("http://localhost:8080/api/audit")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            blocks = data.get("blocks", [])

        print(f"\n📋 Audit Chain: {len(blocks)} blocks")
        print(f"  Chain valid: {data.get('valid', 'unknown')}")

        for block in blocks[-5:]:  # Show last 5 blocks
            print(f"\n  Block #{block['index']}")
            print(f"    Time: {block['timestamp']}")
            print(f"    Operation: {block['operation']}")
            print(f"    Hash: {block['hash'][:32]}...")
            print(f"    Prev:  {block['prev_hash'][:32]}...")

        # Verify chain integrity
        print(f"\n🔗 Chain Integrity: {'✅ VALID' if data.get('valid') else '❌ TAMPERED'}")

    except Exception as e:
        print(f"  ⚠️  Could not fetch audit log: {e}")
        print("  (Make sure the server is running with --server)")


def demonstrate_dashboard():
    """Show dashboard info."""
    print("\n" + "=" * 60)
    print("🌐 Web Dashboard")
    print("=" * 60)
    print(f"\n  Dashboard URL: http://localhost:8080")
    print(f"  API Endpoints:")
    print(f"    GET  /api/stats   - Database statistics")
    print(f"    POST /api/search  - Vector search")
    print(f"    POST /api/insert  - Insert vectors")
    print(f"    GET  /api/audit   - Blockchain audit log")
    print(f"\n  Features:")
    print(f"    • Real-time stats (auto-refresh every 5s)")
    print(f"    • Interactive search interface")
    print(f"    • Audit chain viewer with integrity verification")
    print(f"    • Responsive dark theme UI")


def run_demo():
    """Run the full Phase 3 demo."""
    print("🧬 Organoid VectorDB — Phase 3 Demo")
    print("   HNSW Index + Blockchain Audit + Web Dashboard\n")

    # Connect to gRPC
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.VectorDBStub(channel)

    try:
        stub.Stats(pb2.StatsRequest())
    except grpc.RpcError:
        print("❌ Cannot connect to server. Start it first:")
        print("   python3 demo_phase3.py --server")
        sys.exit(1)

    # Generate synthetic organoid features
    features, labels = generate_organoid_features(500, 128)

    # Run benchmarks
    benchmark_hnsw(stub, features, labels)

    # Show audit log
    demonstrate_audit_log(stub)

    # Show dashboard info
    demonstrate_dashboard()

    print("\n" + "=" * 60)
    print("✅ Phase 3 Demo Complete!")
    print("=" * 60)


def run_server():
    """Start the Rust server."""
    binary = os.path.join(os.path.dirname(__file__), 'target', 'release', 'organoid-vectordb')
    if not os.path.exists(binary):
        print("Building server...")
        subprocess.check_call(['cargo', 'build', '--release'],
                              cwd=os.path.dirname(__file__))

    print("🚀 Starting Organoid VectorDB Server...")
    print("   gRPC:  localhost:50051")
    print("   HTTP:  http://localhost:8080")
    print("   Press Ctrl+C to stop\n")

    os.execvp(binary, [binary, '--dimension', '128', '--http-port', '8080'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3 Demo')
    parser.add_argument('--server', action='store_true', help='Start the server')
    parser.add_argument('--demo', action='store_true', help='Run the demo')
    args = parser.parse_args()

    if args.server:
        run_server()
    elif args.demo:
        run_demo()
    else:
        parser.print_help()
