"""
Interactive Working POC Demo for MATP Contribution 2
Author: Shishir Hegde (NITK Surathkal)

Demonstrates:
1. Ingesting raw verbose tool outputs (SQL, Traceback, CLI/Pytest).
2. Background synthesis into compact structured tuples G = (g, t, p, s, sigma) and (S, R, O) triples.
3. Zero-dependency embedding & vector indexing via matp_index_trajectory.
4. Hybrid semantic retrieval (dense cosine + sparse entity matching) with tool & entity filters.
5. Episodic Knowledge Graph synthesis and cross-step entity linking.
6. Ultra-compact working memory session summary generation for LLM agents.
"""

import sys
import json
from src.matp.memory_store import MATPMemoryStore

def run_demo():
    print("=" * 80)
    print("  MATP: Memory-Augmented Tool Policy -- Contribution 2 Working POC Demo")
    print("  Author: Shishir Hegde (Department of Information Technology, NITK)")
    print("=" * 80)
    
    store = MATPMemoryStore(persistence_file="data/matp.memory.json", embedder_dim=256)
    store.trajectories.clear()
    store.embeddings.clear()

    # Step 1: SQL execution
    print("\n[Step 1] Ingesting Tool Execution: Postgres SQL Query (Data Retrieval)")
    raw_sql = json.dumps([
        {"user_id": 101, "username": "alice", "email": "alice@nitk.edu", "role": "admin", "status": "active"},
        {"user_id": 102, "username": "bob", "email": "bob@nitk.edu", "role": "dev", "status": "active"},
        {"user_id": 103, "username": "charlie", "email": "charlie@nitk.edu", "role": "tester", "status": "inactive"}
    ] * 10, indent=2)
    
    t1 = store.matp_index_trajectory(
        step_id=1,
        sub_goal="Query active engineers and role privileges from Postgres database",
        tool_name="sql_query",
        parameters={"query": "SELECT user_id, username, email, role, status FROM users;", "table": "users"},
        status="SUCCESS",
        raw_output=raw_sql
    )
    print(f"  --> Raw Size:       {t1.raw_token_count_est} tokens ({len(raw_sql)} chars)")
    print(f"  --> Distilled Size: {t1.distilled_token_count_est} tokens ({len(t1.to_compact_text())} chars)")
    print(f"  --> Compression:    {(1.0 - t1.distilled_token_count_est/t1.raw_token_count_est)*100:.1f}% reduction")
    print(f"  --> Entities:       {t1.entities}")
    print(f"  --> Triples (S,R,O):{t1.triples}")
    print(f"  --> Stored Tuple G: {t1.to_compact_text()}")

    # Step 2: Runtime traceback failure
    print("\n[Step 2] Ingesting Tool Execution: Python Migration Script Crash (Failure)")
    raw_tb = (
        "Traceback (most recent call last):\n"
        '  File "/app/scripts/migrate.py", line 42, in <module>\n'
        '    engine.connect()\n'
        "sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at '127.0.0.1', port 5432 failed: Connection refused.\n"
        "Is the server running on that host and accepting TCP/IP connections?\n" * 6
    )
    t2 = store.matp_index_trajectory(
        step_id=2,
        sub_goal="Run database migration scripts on port 5432",
        tool_name="python_interpreter",
        parameters={"script": "scripts/migrate.py"},
        status="FAILURE",
        raw_output=raw_tb
    )
    print(f"  --> Raw Size:       {t2.raw_token_count_est} tokens")
    print(f"  --> Distilled Size: {t2.distilled_token_count_est} tokens")
    print(f"  --> Compression:    {(1.0 - t2.distilled_token_count_est/t2.raw_token_count_est)*100:.1f}% reduction")
    print(f"  --> Entities:       {t2.entities}")
    print(f"  --> Triples (S,R,O):{t2.triples}")
    print(f"  --> Stored Tuple G: {t2.to_compact_text()}")

    # Step 3: Error Remediation & Test Runner
    print("\n[Step 3] Ingesting Tool Execution: Database Service Restart & Pytest Suite (Remediation)")
    raw_test = (
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.11.8, pytest-7.4.3, pluggy-1.3.0\n"
        "rootdir: /app, configfile: pytest.ini\n"
        "collected 12 items\n\n"
        "tests/test_db.py ............                                            [100%]\n"
        "============================== 12 passed in 0.45s =============================="
    )
    t3 = store.matp_index_trajectory(
        step_id=3,
        sub_goal="Start Postgres daemon and verify database integration tests pass",
        tool_name="pytest",
        parameters={"cmd": "pytest tests/test_db.py"},
        status="SUCCESS",
        raw_output=raw_test,
        parent_step_id=2  # Explicit causal link: resolves Step 2 error
    )
    print(f"  --> Raw Size:       {t3.raw_token_count_est} tokens")
    print(f"  --> Distilled Size: {t3.distilled_token_count_est} tokens")
    print(f"  --> Compression:    {(1.0 - t3.distilled_token_count_est/t3.raw_token_count_est)*100:.1f}% reduction")
    print(f"  --> Parent Step:    Step {t3.parent_step_id} (Causal Error Recovery)")
    print(f"  --> Triples (S,R,O):{t3.triples}")
    print(f"  --> Stored Tuple G: {t3.to_compact_text()}")

    # Step 4: Hybrid Querying & Multi-Modal Retrieval
    print("\n" + "=" * 80)
    print("  HYBRID SEMANTIC RETRIEVAL (Dense Cosine + Sparse Entity Matching)")
    print("=" * 80)
    
    test_cases = [
        {"desc": "Semantic Search", "query": "Find the database migration failure exception and connection error", "kwargs": {}},
        {"desc": "Tool-Filtered Query", "query": "database execution results", "kwargs": {"filter_tool": "sql_query"}},
        {"desc": "Entity-Constrained Query", "query": "table user columns", "kwargs": {"filter_entity": "users"}},
    ]

    for tc in test_cases:
        print(f"\n[Test: {tc['desc']}] Query: \"{tc['query']}\" {tc['kwargs']}")
        results = store.matp_query_trajectory(tc["query"], top_k=2, **tc["kwargs"])
        for rank, (res, score) in enumerate(results, 1):
            tpl = res["tuple"]
            print(f"  Rank #{rank} (Hybrid Score = {score:.4f}):")
            print(f"    Step ID: {tpl['step_id']} | Tool: {tpl['tool_name']} | Status: {tpl['status']}")
            print(f"    Summary: {tpl['output_summary']}")
            print(f"    Entities: {tpl.get('entities')}")

    # Step 5: Synthesized Episodic Knowledge Graph
    print("\n" + "=" * 80)
    print("  EPISODIC KNOWLEDGE GRAPH SYNTHESIS (S, R, O Relational Network)")
    print("=" * 80)
    graph = store.get_knowledge_graph()
    print(f"  Total Unique Entities (Nodes): {graph['total_nodes']}")
    print(f"  Total Relational Triples (Edges): {graph['total_edges']}")
    print(f"  Central Hub Entities: {graph['central_entities']}")
    print("  Sample Relational Edges:")
    for edge in graph["edges"][:6]:
        print(f"    ({edge['subject']}) --[{edge['relation']}]--> ({edge['object']}) [Step {edge['step_id']}]")

    # Step 6: Session Working Memory Summary
    print("\n" + "=" * 80)
    print("  LLM WORKING MEMORY SESSION SNAPSHOT (Zero-Overhead Re-population)")
    print("=" * 80)
    print(store.get_session_summary())

    stats = store.get_stats()
    print("\n" + "=" * 80)
    print(f"  SUMMARY: {stats['total_entries']} entries | {stats['total_raw_tokens']} raw tokens -> {stats['total_distilled_tokens']} distilled tokens ({stats['compression_ratio_pct']}% saved | {stats['token_reduction_factor']}x reduction)")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
