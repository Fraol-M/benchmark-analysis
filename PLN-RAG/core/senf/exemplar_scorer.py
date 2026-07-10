from typing import List
from .models import SENF, ExemplarScore
from .exemplar_registry import EXEMPLAR_REGISTRY, CUES


class ExemplarScorer:
    """
    Evaluates context to shift the active exemplar prototype for entities.
    """

    def score(self, senf: SENF, text: str) -> SENF:
        text_lower = text.lower()
        
        for entity in senf.entities:
            if entity.kind and entity.kind.lower() in EXEMPLAR_REGISTRY:
                kind_lower = entity.kind.lower()
                
                # Check for lexical cues in the chunk text
                for cue, updates in CUES.items():
                    if cue in text_lower:
                        for exemplar, score_adj in updates.items():
                            if exemplar in EXEMPLAR_REGISTRY[kind_lower]:
                                senf.exemplars.append(
                                    ExemplarScore(
                                        entity_id=entity.id,
                                        kind=entity.kind,
                                        exemplar=exemplar,
                                        distance=round(1.0 + score_adj, 2),
                                    )
                                )
        return senf
