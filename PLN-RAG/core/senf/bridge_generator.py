import math
from typing import List
from .models import SENF, IdentityEdge, TransWeave


class BridgeGenerator:
    """
    Generates formal PLN logic atoms (SimilarityLink, ContextLink) 
    from the structured SENF layer and its identity graph.
    """

    def generate(
        self, 
        senf: SENF, 
        identity_edges: List[IdentityEdge], 
        weaves: List[TransWeave]
    ) -> List[str]:
        bridge_atoms = []
        entities = {entity.id: entity for entity in senf.entities}
        
        # Process Identity Links
        for edge in identity_edges:
            # Emits a similarity link only if there is sufficient positive evidence 
            # and no strong negative evidence.
            if edge.c_plus <= 0.6 and edge.c_minus >= 0.8:
                
                # Use canonical surface symbols for PeTTa-friendly constants.
                # This path is disabled by default until bridge semantics are tested.
                left_entity = entities.get(edge.left)
                right_entity = entities.get(edge.right)
                left_term = (
                    left_entity.canonical_text
                    if left_entity and left_entity.canonical_text
                    else edge.left
                )
                right_term = (
                    right_entity.canonical_text
                    if right_entity and right_entity.canonical_text
                    else edge.right
                )
                
                # Degrade the truth value based on the identity transport cost
                # Follows paper principle: higher TransWeave/Identity cost = lower truth strength
                # Using the exponential penalty logic: s' = s * exp(-\lambda C(W)), w' = w / (1 + \lambda C(W))
                
                # We calculate the combined identity cost (C_id) as shown in the paper
                # C_id = w+ * c_plus + w- * (1 / (1 + c_minus))
                # For this bridge, we assume lambda=1.0, w+=1.0, w-=1.0, and base s=1.0, w=1.0
                p_minus = 1.0 / (1.0 + edge.c_minus)
                c_id = edge.c_plus + p_minus
                
                stv_s = round(math.exp(-c_id), 3)
                stv_c = round(1.0 / (1.0 + c_id), 3)
                
                bridge_atoms.append(
                    f"(: {left_term}_{right_term}_similarity (SimilarityLink {left_term} {right_term}) (STV {stv_s} {stv_c}))"
                )
                
        # (Future implementation point: process TransWeave links into ContextLinks)
                
        return bridge_atoms
