from daao.ext.maas.scripts.optimized.HumanEval.train.template.operator import (
    Generate,
    GenerateCoT,
    MultiGenerateCoT,
    ScEnsemble,
    Test,
    SelfRefine,
)

operator_mapping = {
    "Generate": Generate,
    "GenerateCoT": GenerateCoT,
    "MultiGenerateCoT": MultiGenerateCoT,
    "ScEnsemble": ScEnsemble,
    "Test": Test,
    "SelfRefine": SelfRefine,
}

operator_names = list(operator_mapping.keys())
