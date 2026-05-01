from typing import Any
from hyperon import MeTTa


def validate_metta_atoms(atoms: list[object]) -> tuple[MeTTa, dict[str, Any]]:
    """
    Validate generated atoms by reparsing their text form and loading them into a
    fresh MeTTa runner.
    """
    parser = MeTTa()
    runner = MeTTa()
    errors: list[dict[str, Any]] = []
    parsed_count = 0
    loaded_count = 0

    for index, atom in enumerate(atoms):
        atom_text = str(atom)

        try:
            parser.parse_single(atom_text)
            parsed_count += 1
        except Exception as exc:
            errors.append({
                "stage": "parse",
                "atom_index": index,
                "atom": atom_text,
                "message": str(exc),
            })
            continue

        try:
            runner.run(atom_text)
            loaded_count += 1
        except Exception as exc:
            errors.append({
                "stage": "load",
                "atom_index": index,
                "atom": atom_text,
                "message": str(exc),
            })

    validation = {
        "valid": len(errors) == 0,
        "syntax_ok": parsed_count == len(atoms),
        "load_ok": loaded_count == len(atoms),
        "atom_count": len(atoms),
        "parsed_count": parsed_count,
        "loaded_count": loaded_count,
        "errors": errors,
    }
    return runner, validation
