from typing import Dict, Literal, Tuple
from daao.ext.maas.benchmark.benchmark import BaseBenchmark
from daao.ext.maas.benchmark.gsm8k import GSM8KBenchmark
from daao.ext.maas.benchmark.humaneval import HumanEvalBenchmark
from daao.ext.maas.benchmark.math import MATHBenchmark

DatasetType = Literal["HumanEval", "GSM8K", "MATH"]


class Evaluator:
    def __init__(self, eval_path: str, batch_size: int):
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
        }

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False
    ):
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")

        data_path = self._get_data_path(dataset, is_test)
        benchmark_class = self.dataset_configs[dataset]

        benchmark = benchmark_class(
            name=dataset, 
            file_path=data_path, 
            log_path=path, 
            batch_size=self.batch_size,
            controller=params["controller"],
            operator_embeddings=params["operator_embeddings"],
            optimizer=params["optimizer"],
        )
        configured_graph = await self._configure_graph(dataset, graph, params)
        if is_test:
            va_list = None 
        else:
            va_list = None
        return await benchmark.run_evaluation(configured_graph, va_list, is_test, params["sample"], params["is_textgrad"])

    async def _configure_graph(self, dataset, graph, params: dict):
        controller = params.get("controller")
        operator_embeddings = params.get("operator_embeddings")
        llm_config = params.get("execute_llm_config")
        dataset_config = params.get("dataset")
        llm_embeddings = params.get("llm_embeddings")
        configured_graph = graph(
            name=dataset,
            llm_config=llm_config,
            dataset=dataset_config,
            controller=controller,
            operator_embeddings=operator_embeddings,
            llm_embeddings = llm_embeddings
        )
        return configured_graph

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"daao/ext/maas/data/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_train.jsonl" 
