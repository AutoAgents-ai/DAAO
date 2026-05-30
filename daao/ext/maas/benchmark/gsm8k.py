import asyncio
import threading
import time
import torch
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from daao.ext.maas.benchmark.benchmark import BaseBenchmark
from daao.logs import logger
from daao.utils.sanitize import sanitize

class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, 
                name: str,
                file_path: str,
                log_path: str,
                batch_size: int,
                controller: torch.nn.Module,
                operator_embeddings: List[List[float]],
                optimizer: torch.optim.Optimizer,):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)

    def extract_number(self, text: str) -> Optional[float]:
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        if prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await asyncio.wait_for(graph(input_text), timeout=1500)

    async def evaluate_problem(self, problem: dict, graph: Callable):
        input_text = problem["question"]
        expected_output = self.extract_number(problem["answer"])

        try:
            output, cost, logprob, vae = await self._generate_output(graph, input_text)
            print("能出来workflow?")
            if not output:
                raise ValueError("output is empty")            

            predicted_number = self.extract_number(output)
            score, extracted_output = self.calculate_score(expected_output, predicted_number)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)
                vae["is_solved"] = 0
            else:
                vae["is_solved"] = 1

            return input_text, output, expected_output, score, cost, logprob, vae

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            vae = {
                    "z_difficulty": torch.zeros((1, 32), device=self.device),
                   "difficulty_scalar":torch.tensor(0.5, device=self.device),
                   "mu":torch.zeros((1, 32), device=self.device),
                   "logvar": torch.zeros((1, 32), device=self.device),
                   "is_solved": 0
            }
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device), vae

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost", "logprob","vae"]
