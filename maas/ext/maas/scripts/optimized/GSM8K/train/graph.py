import torch
import maas.ext.maas.scripts.optimized.GSM8K.train.template.prompt as prompt_custom
import maas.ext.maas.scripts.optimized.GSM8K.train.template.operator as operator
from maas.ext.maas.scripts.optimized.GSM8K.train.template.operator_registry import operator_mapping, operator_names
from maas.provider.llm_provider_registry import create_llm_instance
from maas.utils.cost_manager import CostManager, TokenCostManager
from maas.logs import logger
from maas.configs.models_config import ModelsConfig

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset,
        controller: torch.nn.Module,
        operator_embeddings,
        llm_embeddings
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.cost_manager = CostManager()
        self.custom = operator.Generate(self.llm)
        self.llm_names = ["gpt-4o-mini", "qwen2.5:7b", "gpt-4o", "llama3.1:70b"]
        self.llm_embeddings = llm_embeddings.to(self.device)
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        # self.selection_operator_instances = {
        #     operator_name: operator_mapping[operator_name](self.llm)
        #     for operator_name in operator_names
        # }
        self.selection_operator_names = operator_names

    def llm_ins(self, llm_name):
        models_config = ModelsConfig.default()
        exec_llm_config = models_config.get(llm_name)
        return create_llm_instance(exec_llm_config)
    
    async def __call__(self, problem: str):
        log_probs_layers, selected_names_layers, selected_llms_layers, z_difficulty, difficulty_scalar, mu, logvar = self.controller.forward(
            problem,
            self.operator_embeddings,
            self.llm_embeddings,
            self.selection_operator_names
        )
        print("===================llm")
        print(selected_llms_layers)
        print("==================operator")
        print(selected_names_layers)
        vae = {
            "z_difficulty": z_difficulty,
            "difficulty_scalar": difficulty_scalar,
            "mu": mu,
            "logvar": logvar
        }
        # 实例化所有llm
        llm_instance = {"gpt-4o-mini": self.llm}
        for llm_name in self.llm_names:
            if llm_name == "gpt-4o-mini":
                continue
            if "gpt" in llm_name:
                llm = self.llm_ins(llm_name)
                llm.cost_manager = CostManager()
            else:
                llm = self.llm_ins(llm_name)
                llm.cost_manager = TokenCostManager()
            llm_instance[llm_name] = llm
        print("==========================llm_instance")
        print(llm_instance)


        current_solution = "" 
        solutions = []
        sum_log_prob = 0.0
        total_cost = 0

        for i, (ops, llms) in enumerate(zip(selected_names_layers, selected_llms_layers)):
            assert len(ops) == len(llms), f"第 {i} 层 selected_names 和 selected_llms 数量不一致！"

        for layer_idx, selected_names in enumerate(selected_names_layers):
            for op_idx, op_name in enumerate(selected_names):
                
                try:
                    llm_name = self.llm_names[selected_llms_layers[layer_idx][op_idx]]
                except IndexError:
                    print(f"[警告] 索引越界: selected_llms_layers[{layer_idx}][{op_idx}]")
                    llm_name = "gpt-4o-mini"

                selected_operator = operator_mapping[op_name](llm_instance[llm_name])

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

        for key, value in llm_instance.items():
            print("============================llm cost")
            print(value.cost_manager.total_cost)
            total_cost += value.cost_manager.total_cost
        
        print("能成功出去")
        print(total_cost)
        print(vae)
        if verification['output'] and verification['output'] != "No code generated":
            return verification['output'], total_cost, sum_log_prob, vae
        else:
            return final_solution, total_cost, sum_log_prob, vae
