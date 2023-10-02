import dataclasses


@dataclasses.dataclass
class IQLConfig:
    learning_rate: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.8
    temperature: float = 0.1
