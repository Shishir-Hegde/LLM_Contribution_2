"""
MATP Model Context Protocol (MCP) Server for Contribution 2.
Exposes matp_index_trajectory and matp_query_trajectory over stdio JSON-RPC.
"""

import sys
import json
from typing import Dict, Any
from .memory_store import MATPMemoryStore

def handle_rpc():
    store = MATPMemoryStore(persistence_file="data/matp.memory.json", embedder_dim=256)
    
    # Process line-by-line stdio JSON-RPC
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "matp_index_trajectory",
                                "description": "Synthesizes and indexes raw tool outputs into compact semantic memory buffer G with relational triples.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "step_id": {"type": "integer"},
                                        "sub_goal": {"type": "string"},
                                        "tool_name": {"type": "string"},
                                        "parameters": {"type": "object"},
                                        "status": {"type": "string"},
                                        "raw_output": {"type": "string"},
                                        "parent_step_id": {"type": "integer"}
                                    },
                                    "required": ["step_id", "sub_goal", "tool_name", "parameters", "status", "raw_output"]
                                }
                            },
                            {
                                "name": "matp_query_trajectory",
                                "description": "Queries historical distilled trajectory records using hybrid dense-sparse semantic retrieval.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "top_k": {"type": "integer", "default": 3},
                                        "filter_status": {"type": "string"},
                                        "filter_tool": {"type": "string"},
                                        "filter_entity": {"type": "string"}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "matp_inspect_graph",
                                "description": "Returns the accumulated episodic Knowledge Graph (nodes, directed triples, and key entities) across the session.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "entity_filter": {"type": "string", "description": "Optional entity to filter relational triples by"}
                                    }
                                }
                            },
                            {
                                "name": "matp_get_context_summary",
                                "description": "Generates a compact, high-density session trajectory summary (<250 tokens) for LLM working-memory recovery.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})
                if tool_name == "matp_index_trajectory":
                    distilled = store.matp_index_trajectory(
                        step_id=args.get("step_id", 0),
                        sub_goal=args.get("sub_goal", ""),
                        tool_name=args.get("tool_name", ""),
                        parameters=args.get("parameters", {}),
                        status=args.get("status", "SUCCESS"),
                        raw_output=args.get("raw_output", ""),
                        parent_step_id=args.get("parent_step_id")
                    )
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(distilled.to_dict())}]
                        }
                    }
                elif tool_name == "matp_query_trajectory":
                    results = store.matp_query_trajectory(
                        query=args.get("query", ""),
                        top_k=args.get("top_k", 3),
                        filter_status=args.get("filter_status"),
                        filter_tool=args.get("filter_tool"),
                        filter_entity=args.get("filter_entity")
                    )
                    formatted = [
                        {"score": s, "trajectory": r["tuple"]}
                        for r, s in results
                    ]
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(formatted, indent=2)}]
                        }
                    }
                elif tool_name == "matp_inspect_graph":
                    ent_filter = args.get("entity_filter")
                    if ent_filter:
                        graph_data = store.query_knowledge_graph(ent_filter)
                    else:
                        graph_data = store.get_knowledge_graph()
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(graph_data, indent=2)}]
                        }
                    }
                elif tool_name == "matp_get_context_summary":
                    summary_text = store.get_session_summary()
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": summary_text}]
                        }
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Tool '{tool_name}' not found"}
                    }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"status": "ok"}
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(e)}
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    handle_rpc()
