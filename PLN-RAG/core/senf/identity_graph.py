from typing import List
from .models import SENF, IdentityEdge, Entity


class IdentityGraphBuilder:
    """
    Builds costed identity relationships between entities in nearby chunks.
    """

    def build(self, current_senf: SENF, previous_senfs: List[SENF]) -> List[IdentityEdge]:
        edges = []
        all_entities = list(current_senf.entities)
        for ps in previous_senfs:
            all_entities.extend(ps.entities)

        for i, e1 in enumerate(all_entities):
            for j, e2 in enumerate(all_entities):
                if i >= j:
                    continue
                
                edge = self.score_identity(e1, e2)
                if edge:
                    edges.append(edge)
        return edges

    def score_identity(self, e1: Entity, e2: Entity) -> IdentityEdge | None:
        # Default high cost for identification (c_plus) and low cost for refutation (c_minus)
        c_plus = 2.0
        c_minus = 0.1
        reasons_plus = []
        reasons_minus = []

        if e1.text and e2.text:
            text1 = e1.text.lower()
            text2 = e2.text.lower()
            if text1 == text2:
                c_plus = min(c_plus, 0.15)  # Very low cost to identify identical text
                reasons_plus.append("lexical-match")
            elif text1 in ["he", "she", "it", "they"] or text2 in ["he", "she", "it", "they"]:
                c_plus = min(c_plus, 0.35)  # Moderate cost for pronoun resolution
                reasons_plus.append("pronoun-resolution-candidate")

        if e1.kind and e1.kind == e2.kind:
            c_plus = min(c_plus, c_plus - 0.2 if c_plus < 2.0 else 0.5)
            c_minus = max(c_minus, 0.8) # Hard to refute if kinds match
            reasons_plus.append("same-kind")

        if e1.kind and e2.kind and e1.kind != e2.kind:
            c_plus = 2.0 # High cost to identify different kinds
            c_minus = 0.05 # Very cheap to refute
            reasons_minus.append("different-kind")

        c_plus = max(round(c_plus, 2), 0.05)
        c_minus = max(round(c_minus, 2), 0.05)

        # Only emit edge if there is some positive evidence (cost < 1.0)
        if c_plus < 1.0:
            return IdentityEdge(
                left=e1.id,
                right=e2.id,
                c_plus=c_plus,
                c_minus=c_minus,
                reasons_plus=reasons_plus,
                reasons_minus=reasons_minus,
            )
        return None
