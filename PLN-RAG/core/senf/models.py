from dataclasses import dataclass, field
from typing import Any


@dataclass
class Frame:
    id: str
    head: str
    source_span: str | None = None


@dataclass
class Entity:
    id: str
    kind: str | None = None
    text: str | None = None
    canonical_text: str | None = None
    mention_index: int | None = None
    frame_id: str | None = None
    argument_index: int | None = None
    sentence_id: str | None = None
    chunk_id: str | None = None
    source_id: str | None = None


@dataclass
class Role:
    frame_id: str
    role: str
    entity_id: str


@dataclass
class ExemplarScore:
    entity_id: str
    kind: str
    exemplar: str
    distance: float


@dataclass
class IdentityEdge:
    left: str
    right: str
    c_plus: float
    c_minus: float
    reasons_plus: list[str] = field(default_factory=list)
    reasons_minus: list[str] = field(default_factory=list)
    guard: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class TransWeave:
    source_senf_id: str
    target_senf_id: str
    entity_map: dict[str, str]
    frame_map: dict[str, str]
    kind_map: dict[str, str]
    exemplar_map: dict[str, str]
    role_map: dict[str, str] = field(default_factory=dict)
    time_map: dict[str, str] = field(default_factory=dict)
    loc_map: dict[str, str] = field(default_factory=dict)
    cost: float = 0.0
    distortion: float = 0.0
    reasons: list[str] = field(default_factory=list)
    guarded: bool = False

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class SENF:
    frames: list[Frame] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    roles: list[Role] = field(default_factory=list)
    exemplars: list[ExemplarScore] = field(default_factory=list)
    identity_edges: list[IdentityEdge] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    raw_atoms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
