import re
from typing import Any, List
from .models import SENF, Frame, Entity, Role


class SENFBuilder:
    """
    Builds a structured SENF representation from canonical PLN atoms.
    """

    def __init__(self):
        pass

    def build(self, atoms: List[str], text: str = "", metadata: dict[str, Any] = None) -> SENF:
        senf = SENF(raw_atoms=atoms)
        metadata = metadata or {}
        chunk_id = metadata.get("chunk_id", "unknown_chunk")

        # Fallback counter for unique IDs
        frame_counter = 0
        mention_counter = 0
        kinds_by_text: dict[str, str] = {}

        for atom in atoms:
            # We are looking for simple facts wrapped in STV:
            # e.g., (: fact_name (PredicateName arg1 arg2) (STV 1.0 1.0))
            # We will ignore implications and complex rules for this deterministic v1
            match = re.search(r'\(\:\s+\S+\s+\((.*?)\)\s+\(STV', atom)
            if not match:
                continue

            content = match.group(1).strip()

            if content.startswith("Implication") or content.startswith("Not") or \
               content.startswith("And") or content.startswith("Or") or content.startswith("Premises") or content.startswith("Conclusions"):
                continue

            tokens = content.split()
            if not tokens:
                continue

            predicate = tokens[0]
            args = tokens[1:]

            frame_id = f"f_{frame_counter}"
            frame_counter += 1

            frame = Frame(id=frame_id, head=predicate, source_span=text)
            senf.frames.append(frame)

            for i, arg in enumerate(args):
                if arg.startswith("$"):  # Skip variables
                    continue
                
                # Strip parentheses or extra characters if present
                arg_clean = arg.strip("()")

                # Special case: IsA relations define kinds
                kind = None
                if predicate == "IsA" and i == 0 and len(args) == 2:
                    kind = args[1].strip("()")
                    kinds_by_text[arg_clean] = kind
                elif arg_clean in kinds_by_text:
                    kind = kinds_by_text[arg_clean]

                ent_id = f"{chunk_id}_{frame_id}_m{mention_counter}"
                mention_counter += 1
                senf.entities.append(Entity(
                    id=ent_id,
                    kind=kind,
                    text=arg_clean,
                    canonical_text=self._canonical_text(arg_clean),
                    mention_index=mention_counter - 1,
                    frame_id=frame_id,
                    argument_index=i,
                    chunk_id=chunk_id,
                ))

                role_name = f"arg{i}"
                if predicate == "IsA":
                    role_name = "instance" if i == 0 else "category"

                senf.roles.append(Role(
                    frame_id=frame_id,
                    role=role_name,
                    entity_id=ent_id
                ))

        return senf

    def _canonical_text(self, value: str) -> str:
        clean = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
        clean = clean.replace("-", "_")
        clean = re.sub(r"[^A-Za-z0-9_]", "_", clean)
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean.lower()
