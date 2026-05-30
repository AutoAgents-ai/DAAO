import torch
import daao.ext.maas.scripts.optimized.GSM8K.test.template.prompt as prompt_custom
import daao.ext.maas.scripts.optimized.GSM8K.test.template.operator as operator
from daao.ext.maas.scripts.optimized.GSM8K.test.template.operator_registry import operator_mapping, operator_names
from daao.provider.llm_provider_registry import create_llm_instance
from daao.utils.cost_manager import CostManager
from daao.logs import logger

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset,
        controller: torch.nn.Module,
        operator_embeddings,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.cost_manager = CostManager()
        self.custom = operator.Generate(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }
        self.selection_operator_names = operator_names
        
    async def __call__(self, problem: str):
        log_probs_layers, selected_names_layers = self.controller.forward(problem, self.operator_embeddings, self.selection_operator_names)
        
        current_solution = "" 
        solutions = []
        sum_log_prob = 0.0

        for layer_idx, selected_names in enumerate(selected_names_layers):
            for op_name in selected_names:
                selected_operator = self.selection_operator_instances[op_name]

                if op_name in ["Generate", "GenerateCoT"]:
                    result = await selected_operator(input=problem, instruction=prompt_custom.MATH_SOLVE_PROMPT)
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "SelfRefine":
                    result = await selected_operator(problem=problem, solution=current_solution)
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "Programmer":
                    result = await selected_operator(problem=problem, analysis=current_solution)
                    refined_solution = await self.custom(input=problem + f"\nCode output: {result['code']}", instruction=prompt_custom.REFINE_ANSWER_PROMPT)

                    new_solution = refined_solution['response']
                    solutions.append(new_solution)
                elif op_name == "ScEnsemble":
                    result = await selected_operator(problem=problem, solutions=solutions)
                    solutions = []
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)      
                elif op_name == "MultiGenerateCoT":
                    result = await selected_operator(input=problem,  instruction=prompt_custom.MATH_SOLVE_PROMPT)
                    if isinstance(result, dict) and 'response' in result:
                        for res in result['response']:
                            new_solution = res.get('response', "")
                            solutions.append(new_solution)
                    else:
                        logger.error(f"Expected dict with 'responses' from MultiGenerateCoT, got {type(result)}")
                        new_solution = current_solution
                else:
                    new_solution = current_solution

                current_solution = new_solution

            sum_log_prob += log_probs_layers[layer_idx].item()

        if len(solutions) > 1:
            final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
            final_solution = final_solution['response']
        else:
            final_solution = current_solution
        
        verification = await self.programmer(problem=problem, analysis=final_solution)
        
        if verification['output']:
            return verification['output'], self.llm.cost_manager.total_cost, sum_log_prob
        else:
            return final_solution, self.llm.cost_manager.total_cost, sum_log_prob
