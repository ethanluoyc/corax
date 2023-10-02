import dataclasses


@dataclasses.dataclass(frozen=True)
class LossScalesConfig:
    consistency: float = 2.0
    reward: float = 0.5
    value: float = 0.1
