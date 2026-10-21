"""Model execution choices, independent of hardware and checkpoint storage."""
from dataclasses import asdict, dataclass
import json

from .config import CONFIGS


@dataclass(frozen=True)
class ExecutionPolicy:
    experts: int = 8
    normalization_reference: int = 8
    exit_layer: int = 40

    def __post_init__(self):
        for name in ("experts", "normalization_reference", "exit_layer"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.experts <= 8:
            raise ValueError("--experts must be between 1 and 8")
        if not self.experts <= self.normalization_reference <= 256:
            raise ValueError("--normalization-reference must be between --experts and 256")
        if not 1 <= self.exit_layer <= 40:
            raise ValueError("--exit-layer must be between 1 and 40 (inclusive)")

    @classmethod
    def from_args(cls, args):
        experts = getattr(args, "experts", None)
        if experts is None:
            experts = 4 if getattr(args, "adapter", None) else 8
        return cls(experts, getattr(args, "normalization_reference", 8),
                   getattr(args, "exit_layer", 40))

    def as_dict(self):
        return asdict(self)

    @property
    def custom_routing(self):
        return self.experts != 8 or self.normalization_reference != 8

    @property
    def cache_key(self):
        return f"top{self.experts}-ref{self.normalization_reference}-depth{self.exit_layer}"

    def overrides(self):
        result = {}
        if self.custom_routing:
            result["num_experts_per_tok"] = self.experts
        if self.exit_layer != 40:
            # This config is bundled byte-for-byte from the pinned parent release.
            config = json.loads((CONFIGS / "model-config.json").read_text())
            result.update(num_hidden_layers=self.exit_layer,
                          layer_types=config["layer_types"][:self.exit_layer],
                          clef_unconditional_exit_layer=self.exit_layer)
        return result

