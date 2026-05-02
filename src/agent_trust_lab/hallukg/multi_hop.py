"""Multi-hop anchoring reasoning using NetworkX knowledge graphs.

Extends the single-hop AnchoringReasoner with graph-based multi-hop
path finding. When a triple lacks a direct knowledge match, the
MultiHopReasoner constructs a KnowledgeGraph and searches for
indirect connecting paths.
"""

from typing import Any, Dict, List, Optional, Set

import networkx as nx

from agent_trust_lab.log import get_logger

logger = get_logger("hallukg.multi_hop")


class KnowledgeGraph:
    """A knowledge graph built from triples for multi-hop path finding.

    Nodes are entities (subjects/objects). Edges represent predicates
    connecting entities, with metadata from the source triples.
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self._node_aliases: Dict[str, str] = {}

    @property
    def nodes(self) -> Set[str]:
        return set(self._graph.nodes())

    @property
    def edges(self) -> int:
        return self._graph.number_of_edges()

    def add_triple(self, triple: Dict[str, Any]) -> None:
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("predicate", "")).strip()
        obj = str(triple.get("object", "")).strip()

        if not subject or not obj:
            return

        subj_lower = subject.lower()
        obj_lower = obj.lower()

        extra = {k: v for k, v in triple.items() if k not in ("subject", "predicate", "object")}
        self._graph.add_edge(subj_lower, obj_lower, predicate=predicate, **extra)

    def add_triples(self, triples: List[Dict[str, Any]]) -> None:
        for t in triples:
            self.add_triple(t)

    def add_knowledge_sentence(self, sentence: str) -> None:
        parts = [p.strip() for p in sentence.replace(",", " ").split() if len(p.strip()) >= 3]
        for i in range(len(parts) - 1):
            self._graph.add_edge(
                parts[i].lower(),
                parts[i + 1].lower(),
                predicate="related_to",
                source=sentence,
            )

    def add_knowledge_text(self, text: str) -> None:
        if not text:
            return
        for sentence in text.replace("\n", ". ").split(". "):
            stripped = sentence.strip()
            if stripped:
                self.add_knowledge_sentence(stripped)

    def find_shortest_path(
        self, source: str, target: str, max_hops: int = 3
    ) -> Optional[List[str]]:
        source_lower = source.lower()
        target_lower = target.lower()

        if source_lower == target_lower:
            return [source_lower]

        if source_lower not in self._graph or target_lower not in self._graph:
            return None

        try:
            path = nx.shortest_path(self._graph, source=source_lower, target=target_lower)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        if len(path) - 1 > max_hops:
            return None
        return path

    def get_edge_data(self, u: str, v: str) -> Optional[Dict[str, Any]]:
        try:
            return dict(self._graph.edges[u.lower(), v.lower()])
        except KeyError:
            return None

    def neighbors(self, entity: str) -> List[str]:
        entity_lower = entity.lower()
        if entity_lower not in self._graph:
            return []
        return list(self._graph.neighbors(entity_lower))

    def entity_exists(self, entity: str) -> bool:
        return entity.lower() in self._graph

    def size(self) -> int:
        return self._graph.number_of_nodes()


class MultiHopReasoner:
    """Multi-hop anchoring reasoner using graph-based path finding.

    Attempts to connect extracted triples to knowledge by traversing
    the KnowledgeGraph. If a direct match exists, it acts like single-hop.
    If only indirect paths exist, those paths become multi-hop evidence.
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        grounded_threshold: float = 0.3,
        max_hops: int = 3,
    ):
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.grounded_threshold = grounded_threshold
        self.max_hops = max_hops

    def build_graph_from_triples(self, triples: List[Dict[str, Any]]) -> None:
        self.knowledge_graph.add_triples(triples)

    def build_graph_from_knowledge(self, knowledge_text: str) -> None:
        self.knowledge_graph.add_knowledge_text(knowledge_text)

    def anchor(self, triple: Dict[str, Any], knowledge_text: str = "") -> Dict[str, Any]:
        subject = str(triple.get("subject", ""))
        predicate = str(triple.get("predicate", ""))
        obj = str(triple.get("object", ""))
        confidence = triple.get("confidence", 0.0)

        kg = self.knowledge_graph

        subj_in_kg = kg.entity_exists(subject)
        obj_in_kg = kg.entity_exists(obj)
        both_in_kg = subj_in_kg and obj_in_kg

        if both_in_kg and self._has_edge(subject, obj):
            return self._make_direct_hit(subject, predicate, obj, confidence)

        if both_in_kg:
            path = kg.find_shortest_path(subject, obj, self.max_hops)
            if path and len(path) > 2:
                return self._make_multi_hop_hit(subject, predicate, obj, confidence, path)

        if subj_in_kg:
            neighbors = kg.neighbors(subject)
            for neighbor in neighbors:
                path = kg.find_shortest_path(neighbor, obj, self.max_hops)
                if path and len(path) >= 2:
                    full_path = [subject.lower()] + path
                    return self._make_multi_hop_hit(subject, predicate, obj, confidence, full_path)
                edge = kg.get_edge_data(subject, neighbor)
                if edge and str(edge.get("object", "").lower()) == obj.lower():
                    return self._make_extended_hit(
                        subject,
                        predicate,
                        obj,
                        confidence,
                        neighbor,
                        str(edge.get("predicate", "related_to")),
                    )

        if obj_in_kg:
            for node in kg.nodes:
                neighbors = kg.neighbors(node)
                if obj.lower() in [n.lower() for n in neighbors]:
                    path = kg.find_shortest_path(node, obj, self.max_hops)
                    if path and len(path) >= 2:
                        full_path = [subject.lower()] + path
                        return self._make_multi_hop_hit(
                            subject, predicate, obj, confidence, full_path
                        )
                    break

        knowledge_lower = knowledge_text.lower() if knowledge_text else ""
        if knowledge_lower:
            subj_match = subject.lower() in knowledge_lower
            obj_match = obj.lower() in knowledge_lower
            if subj_match or obj_match:
                score = 0.6 if (subj_match and obj_match) else 0.4
                grounded = score >= self.grounded_threshold
                label = "Grounded" if grounded else "Ungrounded"
                matched = []
                if subj_match:
                    matched.append(f"'{subject}'")
                if obj_match:
                    matched.append(f"'{obj}'")
                return {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence,
                    "label": label,
                    "evidence": [
                        f"Multi-hop: partial match ({', '.join(matched)}) in knowledge source"
                    ],
                    "anchor_score": round(score, 4),
                    "multi_hop": True,
                    "hop_count": 1,
                }

        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "label": "Ungrounded",
            "evidence": [f"Multi-hop: no path found for '{subject}' -> '{obj}' in knowledge graph"],
            "anchor_score": 0.0,
            "multi_hop": True,
            "hop_count": 0,
        }

    def batch_anchor(
        self, triples: List[Dict[str, Any]], knowledge_text: str = ""
    ) -> List[Dict[str, Any]]:
        if not triples:
            return []
        if knowledge_text:
            self.build_graph_from_knowledge(knowledge_text)
        self.build_graph_from_triples(triples)
        results = [self.anchor(t, knowledge_text) for t in triples]
        return results

    def _has_edge(self, source: str, target: str) -> bool:
        return self.knowledge_graph.get_edge_data(source, target) is not None

    def _make_direct_hit(
        self, subject: str, predicate: str, obj: str, confidence: float
    ) -> Dict[str, Any]:
        edge = self.knowledge_graph.get_edge_data(subject, obj) or {}
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "label": "Grounded",
            "evidence": [
                f"Multi-hop: direct edge {subject} -> {obj} "
                f"(predicate: {edge.get('predicate', 'unknown')})"
            ],
            "anchor_score": 0.95,
            "multi_hop": True,
            "hop_count": 1,
        }

    def _make_multi_hop_hit(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float,
        path: List[str],
    ) -> Dict[str, Any]:
        hop_count = len(path) - 1
        score = 1.0 - (0.15 * (hop_count - 1))
        score = max(0.3, score)
        grounded = score >= self.grounded_threshold
        label = "Grounded" if grounded else "Ungrounded"
        path_str = " -> ".join(path)
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "label": label,
            "evidence": [f"Multi-hop path ({hop_count} hops): {path_str}"],
            "anchor_score": round(score, 4),
            "multi_hop": True,
            "hop_count": hop_count,
        }

    def _make_extended_hit(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float,
        intermediate: str,
        intermediate_pred: str,
    ) -> Dict[str, Any]:
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "label": "Grounded",
            "evidence": [
                f"Multi-hop path (2 hops): {subject} --{intermediate_pred}--> "
                f"{intermediate} --related_to--> {obj}"
            ],
            "anchor_score": 0.85,
            "multi_hop": True,
            "hop_count": 2,
        }
