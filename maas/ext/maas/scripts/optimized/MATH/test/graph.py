import torch
import maas.ext.maas.scripts.optimized.MATH.train.template.prompt as prompt_custom
import maas.ext.maas.scripts.optimized.MATH.train.template.operator as operator
from maas.ext.maas.scripts.optimized.MATH.train.template.operator_registry import operator_mapping, operator_names
from maas.provider.llm_provider_registry import create_llm_instance
from maas.utils.cost_manager import CostManager
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
        self.programmer = operator.Programmer(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.llm_names = ["gpt-4o-mini", "qwen2.5:7b", "qwen2.5:7b-instruct", "gpt-4o"]

        self.controller = controller.to(self.device)
        #[num, dim=384]
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.llm_embeddings = llm_embeddings.to(self.device)
        self.selection_operator_names = operator_names
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }

    def llm_ins(self, llm_name):
        models_config = ModelsConfig.default()
        exec_llm_config = models_config.get(llm_name)
        return create_llm_instance(exec_llm_config)

        
    async def __call__(self, problem: str):
        # 控制器 forward 输出
        log_probs_layers, selected_names_layers, selected_llms_layers, z_difficulty, difficulty_scalar, mu, logvar = self.controller.forward(
            problem,
            self.operator_embeddings,
            self.llm_embeddings,
            self.selection_operator_names
        )
        print("===========================================llm")
        print(selected_llms_layers)
        print("============================================operator")
        print(selected_names_layers)
        # VAE 相关记录
        vae = {
            "z_difficulty": z_difficulty,
            "difficulty_scalar": difficulty_scalar,
            "mu": mu,
            "logvar": logvar
        }


        current_solution = ""
        solutions = []
        sum_log_prob = 0.0

        # 初始化一步：代码生成 + 初步 refine
        code_solution = await self.programmer(problem=problem)
        refined_solution = await self.custom(
            input=problem + f"\nCode output: {code_solution['output']}",
            instruction=prompt_custom.REFINE_ANSWER_PROMPT
        )
        solutions.append(refined_solution['response'])

        # 执行 operator pipeline
        for layer_idx, selected_names in enumerate(selected_names_layers):
            for op_idx, op_name in enumerate(selected_names):
                selected_operator = operator_mapping[op_name](self.llm_ins(self.llm_names[selected_llms_layers[layer_idx][op_idx]]))

                if op_name in ["Generate", "GenerateCoT"]:
                    result = await selected_operator(input=problem, instruction=prompt_custom.DETAILED_SOLUTION_PROMPT)
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "SelfRefine":
                    result = await selected_operator(problem=problem, solution=current_solution)
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "Programmer":
                    result = await selected_operator(problem=problem, analysis=current_solution)
                    new_solution = result['output']
                    solutions.append(new_solution)
                elif op_name == "ScEnsemble":
                    result = await selected_operator(problem=problem, solutions=solutions)
                    solutions = []
                    new_solution = result.get('response', "")
                    solutions.append(new_solution)
                elif op_name == "MultiGenerateCoT":
                    result = await selected_operator(input=problem, instruction=prompt_custom.GENERATE_SOLUTION_PROMPT)
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

            # 新版 log_probs_layers 已包含 LLM 选择 log prob
            sum_log_prob += log_probs_layers[layer_idx]

        # 最终整合
        if len(solutions) > 1:
            final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
            final_solution = final_solution['response']
        else:
            final_solution = current_solution

        print("==========================================================can output")
        return final_solution, self.llm.cost_manager.total_cost, sum_log_prob, vae