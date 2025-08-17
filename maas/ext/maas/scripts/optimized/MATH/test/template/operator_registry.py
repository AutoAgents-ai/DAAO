from maas.ext.maas.scripts.optimized.MATH.train.template.operator import (
    Generate,
    GenerateCoT,
    MultiGenerateCoT,
    ScEnsemble,
    Programmer,
    SelfRefine,
)

operator_mapping = {
    "Generate": Generate,
    "GenerateCoT": GenerateCoT,
    "MultiGenerateCoT": MultiGenerateCoT,
    "ScEnsemble": ScEnsemble,
    "Programmer": Programmer,
    "SelfRefine": SelfRefine,
}

operator_names = list(operator_mapping.keys())
