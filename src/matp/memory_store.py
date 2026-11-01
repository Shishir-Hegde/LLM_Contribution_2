"""
Trajectory Memory Buffer and Vector Store for MATP.
Persists and retrieves distilled trajectory tuples G = (sub_goal, tool, parameters, status, output_summary).
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from .abstraction import TrajectoryTuple, BackgroundAbstractionEngine
from .embedder import LocalNGramEmbedder

class MATPMemoryStore:
    """
    Persistent Trajectory Vector Store.
    Implements matp_index_trajectory and matp_query_trajectory.
    """
    def __init__(self, persistence_file: Optional[str] = "data/matp.memory.json", embedder_dim: int = 128):
        self.persistence_file = persistence_file
        self.embedder = LocalNGramEmbedder(dim=embedder_dim)
        self.abstraction_engine = BackgroundAbstractionEngine()
        self.trajectories: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        
        if self.persistence_file and os.path.exists(self.persistence_file):
            self.load()

    def matp_index_trajectory(
        self,
        step_id: int,
        sub_goal: str,
        tool_name: str,
        parameters: Dict[str, Any],
        status: str,
        raw_output: str,
        parent_step_id: Optional[int] = None
    ) -> TrajectoryTuple:
        """
        Synthesizes the raw output, generates vector embedding of distilled tuple G,
        and indexes it into the persistent memory store.
        """
        # 1. Background Abstraction & Trajectory Synthesis
        distilled = self.abstraction_engine.abstract_tool_output(
            step_id=step_id,
            sub_goal=sub_goal,
            tool_name=tool_name,
            parameters=parameters,
            status=status,
            raw_output=raw_output,
            parent_step_id=parent_step_id
        )

        # 2. Embed distilled compact representation
        compact_text = distilled.to_compact_text()
        vector = self.embedder.embed(compact_text)

        # 3. Store record
        entry = {
            "tuple": distilled.to_dict(),
            "compact_text": compact_text,
            "raw_output_tokens": distilled.raw_token_count_est,
            "distilled_tokens": distilled.distilled_token_count_est,
            "indexed_at": time.time()
        }
        self.trajectories.append(entry)
        self.embeddings.append(vector)

        if self.persistence_file:
            self.save()

        return distilled

    def matp_query_trajectory(
        self,
        query: str,
        top_k: int = 3,
        filter_status: Optional[str] = None,
        filter_tool: Optional[str] = None,
        filter_entity: Optional[str] = None,
        hybrid_alpha: float = 0.75
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries the memory store using hybrid semantic retrieval:
        combines dense subword n-gram cosine similarity with sparse entity/keyword overlap,
        with optional tool, status, and entity constraints.
        """
        if not self.trajectories:
            return []

        query_vec = self.embedder.embed(query)
        scored: List[Tuple[Dict[str, Any], float]] = []
        
        # Tokenize query words for sparse entity overlap
        import re
        q_tokens = set(re.findall(r"[a-zA-Z0-9_\-\.]+", query.lower()))

        for idx, (entry, vec) in enumerate(zip(self.trajectories, self.embeddings)):
            tpl = entry["tuple"]
            
            # Status filter
            if filter_status and tpl.get("status") != filter_status:
                continue

            # Tool filter
            if filter_tool and tpl.get("tool_name") != filter_tool:
                continue

            # Entity filter
            if filter_entity:
                fe = filter_entity.lower()
                ent_lower = [str(e).lower() for e in tpl.get("entities", [])]
                if not any(fe in e for e in ent_lower) and fe not in entry["compact_text"].lower():
                    continue

            # Dense cosine similarity
            cos_sim = self.embedder.cosine_similarity(query_vec, vec)
            
            # Sparse entity & keyword overlap
            target_tokens = set(re.findall(r"[a-zA-Z0-9_\-\.]+", entry["compact_text"].lower()))
            for e in tpl.get("entities", []):
                target_tokens.update(re.findall(r"[a-zA-Z0-9_\-\.]+", str(e).lower()))
            
            overlap_count = len(q_tokens.intersection(target_tokens))
            sparse_score = (overlap_count / max(1, len(q_tokens))) if q_tokens else 0.0

            # Hybrid score combination
            norm_dense = max(0.0, cos_sim)
            hybrid_score = (hybrid_alpha * norm_dense) + ((1.0 - hybrid_alpha) * sparse_score)
            
            scored.append((entry, round(hybrid_score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_knowledge_graph(self) -> Dict[str, Any]:
        """
        Synthesizes the episodic Knowledge Graph accumulated across all recorded trajectories.
        Returns unique nodes, directed (Subject, Relation, Object) edges, and entity degrees.
        """
        from collections import Counter
        nodes = set()
        edges = []
        degrees = Counter()

        for entry in self.trajectories:
            tpl = entry["tuple"]
            step_id = tpl["step_id"]
            for s, r, o in tpl.get("triples", []):
                s_str, r_str, o_str = str(s), str(r), str(o)
                nodes.add(s_str)
                nodes.add(o_str)
                degrees[s_str] += 1
                degrees[o_str] += 1
                edges.append({
                    "subject": s_str,
                    "relation": r_str,
                    "object": o_str,
                    "step_id": step_id
                })

        central = [entity for entity, count in degrees.most_common(8)]
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": sorted(list(nodes)),
            "edges": edges,
            "central_entities": central
        }

    def query_knowledge_graph(self, entity: str) -> List[Dict[str, Any]]:
        """
        Searches the episodic Knowledge Graph for all relations involving a specific entity.
        """
        ent_lower = entity.lower()
        graph = self.get_knowledge_graph()
        matched = []
        for edge in graph["edges"]:
            if ent_lower in edge["subject"].lower() or ent_lower in edge["object"].lower():
                matched.append(edge)
        return matched

    def get_session_summary(self) -> str:
        """
        Generates a concise, high-density session summary string (<250 tokens)
        for rapid agent working-memory re-population without context overflow.
        """
        if not self.trajectories:
            return "No trajectories recorded."

        lines = [f"=== MATP Session Trajectory Log ({len(self.trajectories)} steps) ==="]
        for entry in self.trajectories:
            tpl = entry["tuple"]
            lines.append(
                f"Step {tpl['step_id']} [{tpl['status']}] {tpl['tool_name']}: {tpl['output_summary']}"
            )
        stats = self.get_stats()
        lines.append(
            f"Metrics: {stats['total_raw_tokens']} raw tokens distilled to {stats['total_distilled_tokens']} tokens ({stats['compression_ratio_pct']}% saved)."
        )
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        raw_total = sum(e["raw_output_tokens"] for e in self.trajectories)
        distilled_total = sum(e["distilled_tokens"] for e in self.trajectories)
        compression_ratio = (1.0 - (distilled_total / max(1, raw_total))) * 100.0
        return {
            "total_entries": len(self.trajectories),
            "total_raw_tokens": raw_total,
            "total_distilled_tokens": distilled_total,
            "compression_ratio_pct": round(compression_ratio, 2),
            "token_reduction_factor": round(raw_total / max(1, distilled_total), 2)
        }

    def save(self):
        if not self.persistence_file:
            return
        os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
        with open(self.persistence_file, "w", encoding="utf-8") as f:
            json.dump({
                "trajectories": self.trajectories,
                "embeddings": self.embeddings
            }, f, indent=2)

    def load(self):
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return
        try:
            with open(self.persistence_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.trajectories = data.get("trajectories", [])
                self.embeddings = data.get("embeddings", [])
        except Exception as e:
            print(f"Warning: Failed to load existing memory store: {e}")
