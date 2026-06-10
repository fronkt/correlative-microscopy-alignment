from cma.pipeline.classical import ClassicalResult, classical_register
from cma.pipeline.register import RegistrationResult, register
from cma.pipeline.register_v2 import RegistrationV2Result, register_v2
from cma.pipeline.verify import verification_score

__all__ = [
    "ClassicalResult",
    "RegistrationResult",
    "RegistrationV2Result",
    "classical_register",
    "register",
    "register_v2",
    "verification_score",
]
