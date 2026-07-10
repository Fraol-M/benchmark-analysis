from typing import List
from .models import SENF, TransWeave


class TransWeaveBuilder:
    """
    Builds costed alignments between SENF objects to handle weak rule transfer.
    """

    def build_local(self, current_senf: SENF, previous_senfs: List[SENF]) -> List[TransWeave]:
        weaves = []
        if not previous_senfs:
            return weaves
            
        target_senf = previous_senfs[-1]
        
        # v1 MVP: Top-k alignment focusing on exact frame matches
        for sf in current_senf.frames:
            for tf in target_senf.frames:
                if sf.head == tf.head:
                    weaves.append(
                        TransWeave(
                            source_senf_id="current",
                            target_senf_id="previous",
                            entity_map={},
                            frame_map={sf.id: tf.id},
                            kind_map={},
                            exemplar_map={},
                            cost=0.2,
                            distortion=0.1,
                            reasons=["exact-frame-head-match"],
                        )
                    )
        
        return weaves
