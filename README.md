# MATP: Memory-Augmented Tool Policy
## Contribution 2: Background Abstraction & Trajectory Synthesis Engine
**Author:** Shishir Hegde  
**Department of Information Technology, NITK Surathkal**

---

## 1. Problem Statement & Deliverable Objectives Breakdown

In complex agentic LLM workflows spanning long execution horizons, raw outputs returned by terminal commands, SQL databases, REST APIs, and document parsers quickly consume and overwhelm the context window.
Standard context management strategies suffer from two major dilemmas:
1. **FIFO History Limits (e.g. FIFO(3)):** Drastically prunes history, causing **catastrophic forgetting** of essential schemas, entity states, and fixes discovered early in the session.
2. **Unpruned Retention / Naive RAG:** Floods memory with massive raw boilerplate (stack frames, JSON headers, repetitive lines), causing **token bloat**, high inference latency/costs, and **retrieval degradation** due to keyword noise dilution.

### Decomposition into Two Main Deliverable Objectives:

* **Objective 1 (Phase 1 / Mid-Semester Deliverable — Built Working POC):**
  * **Atomic Trajectory Abstraction & Distillation Engine:** Transform arbitrary raw outputs $O_{\text{raw}}$ into compact structured tuples $G = (g, t, p, s, \sigma, \text{parent})$ and relational $(S, R, O)$ triples across 5+ modalities (SQL, tracebacks, REST endpoints, Bash/Git/Pytest, Doc/PDF).
  * **Zero-Dependency Persistent Vector Buffer:** Implement local deterministic subword $n$-gram vectorization with term frequency saturation and single-file cosine similarity storage (`data/matp.memory.json`).
  * **Hybrid Semantic Retrieval:** Combine dense subword cosine similarity with sparse entity/keyword matching and tool/entity filters.
  * **Episodic Knowledge Graph Foundation:** Aggregate directed triples $(S, R, O)$ across steps, identify central hub entities, and track causal error-remediation parent links.
  * **Complete Model Context Protocol (MCP) Interface:** Expose 4 tools (`matp_index_trajectory`, `matp_query_trajectory`, `matp_inspect_graph`, `matp_get_context_summary`) over stdio JSON-RPC.
  * **Empirical Benchmarking & Metrics:** Achieve $93.8\%$ token compression ($16.1\times$ reduction), 100% key entity retention, and 90.0% P@1 retrieval precision.

* **Objective 2 (Phase 2 / End-Semester Extension — Next Steps):**
  * **Cross-Session Causal DAG Resolution:** Invalidation propagation across multi-step mutations and dependent resources.
  * **Hierarchical Macro-Action Distillation:** Cluster recurrent 10-step subroutines into reusable single macro-actions.
  * **Asynchronous Background Distillation Worker:** Offload deep neural summarization (e.g., via local Ollama/Qwen-2.5 models) to a non-blocking background queue.
  * **Multi-Agent Evaluation:** Benchmark against real-world agentic suites (SWE-bench & GAIA).

---

## 2. Directory Structure

```
├── demo.py                          # Interactive CLI demo of Contribution 2 (Graph, Hybrid Search, Summary)
├── src/
│   └── matp/
│       ├── __init__.py
│       ├── abstraction.py           # Core Abstraction Engine with multi-domain regex & triple extraction
│       ├── embedder.py              # Deterministic subword n-gram hash vectorizer
│       ├── memory_store.py          # Trajectory Vector Store, Episodic Graph & Hybrid Search
│       ├── mcp_server.py            # Stdio JSON-RPC MCP Server (4 tools exposed)
│       └── benchmark.py             # Quantitative benchmarking & plot generator
├── figures/
│   ├── fig1_token_compression.png   # Token reduction across 5 domains (O_raw vs G)
│   ├── fig2_context_growth.png      # 30-turn context accumulation (Raw vs FIFO vs MATP)
│   ├── fig3_retrieval_and_latency.png # P@1/P@3 precision, fact recall & latency scaling
│   └── fig4_signal_noise_ratio.png  # Noise filtering vs core signal extraction
├── presentation/
│   └── matp_presentation_updated.tex# Beamer presentation with Contrib 2 slides integrated
└── data/
    └── matp.memory.json             # Persistent trajectory vector database
```

---

## 3. How to Run the POC & Benchmarks

### 1. Run Interactive POC Demo
```bash
python demo.py
```

### 2. Run Comprehensive Benchmarks & Regenerate Figures
```bash
python -m src.matp.benchmark
```

### 3. Run MCP Server over Stdio
```bash
python -m src.matp.mcp_server
```

---

## 4. Key Benchmark Results Summary

| Tool Domain | Raw Tokens ($O_{\text{raw}}$) | Distilled Tokens ($G$) | Compression Ratio | Reduction Factor |
| :--- | :---: | :---: | :---: | :---: |
| **SQL Database** | 2,543 | 160 | **93.7%** | $15.9\times$ |
| **Runtime Tracebacks** | 1,614 | 102 | **93.7%** | $15.8\times$ |
| **REST API Responses** | 1,208 | 166 | **86.3%** | $7.3\times$ |
| **File System / Bash** | 3,277 | 139 | **95.8%** | $23.6\times$ |
| **Doc / PDF Parsing** | 1,904 | 90 | **95.3%** | $21.2\times$ |
| **Overall Average** | **2,109 / call** | **131 / call** | **93.8%** | **16.1$\times$ Reduction** |

* **Context Accumulation (30 turns):** Raw context blows up to **63,276 tokens** (exceeding standard 8k context window), whereas MATP Distilled Buffer retains all 30 turns within **3,940 tokens** ($<4\text{k}$ budget).
* **Retrieval Quality:** MATP achieves **90.0% P@1**, **80.0% P@3**, and **100.0% Key Fact Retention** vs Raw Log Store (**60.0% P@1** and **65.0% retention** due to noise dilution).
* **Scan Latency:** Sub-millisecond CPU execution with zero network lag.
