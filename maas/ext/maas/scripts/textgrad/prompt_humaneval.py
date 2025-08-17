import json

operator_example = {
    "thought": {
        "Insights": "The Generate operator is designed to leverage the LLM's capability to dynamically generate responses based on input prompts. It serves as a core text generation module for agents.",
        "Overall Idea": "This operator abstracts text generation into a reusable component. By accepting an input string and optional mode, it can adapt to different prompting strategies (e.g., XML formatting) while maintaining a simple interface. The asynchronous implementation ensures compatibility with modern AI agent architectures.",
        "Implementation": "1. Inherit from base Operator class\n2. Implement async __call__ method\n3. Integrate with prompt templating\n4. Support multiple response formats through mode parameter\n5. Maintain state isolation through instance-based design"},
    "description": "Core text generation operator that uses LLMs to produce responses based on structured prompts and input data. Supports asynchronous execution and multiple response formats.",
    "code": """
    class Generate(Operator):
        def __init__(self, llm: LLM, name: str = "Generate"):
            super().__init__(llm, name)
            self.response_format = "xml"  # Default format
        
        async def __call__(self, input: str, mode: str = None):
            \"\"\"Generate response using structured prompt template
            Args:
                input: Input text or prompt template variables
                mode: Response format mode (xml/json/text)
            Returns:
                Generated response string
            \"\"\"
            prompt = GENERATE_PROMPT.format(input=input)
            response_mode = mode or self.response_format
            response = await self._fill_node(GenerateOp, prompt, mode=response_mode)
            return self._postprocess(response)
        
        def _postprocess(self, raw_response: str) -> str:
            \"\"\"Clean and validate generated response\"\"\"
            return raw_response.strip().replace("\\n\\n", "\\n")
"""
}

Generate = {
    "thought": {
        "Insights": "The Generate operator is designed to leverage the LLM's capability to dynamically generate responses based on input prompts. It serves as a core text generation module for agents.",
        "Overall Idea": "This operator abstracts text generation into a reusable component. By accepting an input string and optional mode, it can adapt to different prompting strategies (e.g., XML formatting) while maintaining a simple interface. The asynchronous implementation ensures compatibility with modern AI agent architectures.",
        "Implementation": "1. Inherit from base Operator class\n2. Implement async __call__ method\n3. Integrate with prompt templating\n4. Support multiple response formats through mode parameter\n5. Maintain state isolation through instance-based design"},
    "description": "Core text generation operator that uses LLMs to produce responses based on structured prompts and input data. Supports asynchronous execution and multiple response formats.",
    "code": """
    class Generate(Operator):
        def __init__(self, llm: LLM, name: str = "Generate"):
            super().__init__(llm, name)
            self.response_format = "xml"  # Default format
        
        async def __call__(self, input: str, mode: str = None):
            \"\"\"Generate response using structured prompt template
            Args:
                input: Input text or prompt template variables
                mode: Response format mode (xml/json/text)
            Returns:
                Generated response string
            \"\"\"
            prompt = GENERATE_PROMPT.format(input=input)
            response_mode = mode or self.response_format
            response = await self._fill_node(GenerateOp, prompt, mode=response_mode)
            return self._postprocess(response)
        
        def _postprocess(self, raw_response: str) -> str:
            \"\"\"Clean and validate generated response\"\"\"
            return raw_response.strip().replace("\\n\\n", "\\n")
"""
}

GenerateCoT = {
    "thought": {
        "Insights": "Enhances reasoning transparency through explicit chain-of-thought generation",
        "Overall Idea": "Structured CoT generation for complex problem solving with traceable logic steps",
        "Implementation": "1. Specialized CoT prompt template\n2. XML response parsing\n3. Async-compatible execution"
    },
    "description": "Chain-of-Thought generator that produces explicit reasoning steps using structured XML output format",
    "code": """
    class GenerateCoT(Operator):
        def __init__(self, llm: LLM, name: str = "GenerateCoT"):
            super().__init__(llm, name)
            self.cot_cache = {}

        async def __call__(self, input: str, mode: str = None):
            \"\"\"Generate structured reasoning chain for complex problems
            Args:
                input: Problem description with solution requirements
                mode: Output structure format (default: XML)
            Returns:
                Dictionary with 'reasoning_chain' and 'final_answer' keys
            \"\"\"
            prompt = GENERATECOT_PROMPT.format(input=input)
            response = await self._fill_node(GenerateCoTOp, prompt, mode="xml_fill")
            return self._parse_cot(response)

        def _parse_cot(self, raw_response: dict) -> dict:
            \"\"\"Extract reasoning steps from XML response\"\"\"
            return {
                'reasoning_chain': raw_response.get('steps', []),
                'final_answer': raw_response.get('conclusion', '')
            }
"""
}

MultiGenerateCoT = {
    "thought": {
        "Insights": "Diverse reasoning paths improve solution robustness",
        "Overall Idea": "Parallel generation of multiple distinct reasoning chains",
        "Implementation": "1. Concurrent CoT generation\n2. Result aggregation\n3. Diversity sampling"
    },
    "description": "Generates multiple independent reasoning chains for solution verification and ensemble analysis",
    "code": """
    class MultiGenerateCoT(Operator):
        def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
            super().__init__(llm, name)
            self.max_attempts = 3

        async def __call__(self, input: str, mode: str = None):
            \"\"\"Produce multiple reasoning paths for same input problem
            Args:
                input: Target problem statement
                mode: Consistency mode for parallel generations
            Returns:
                List of dictionaries with varied reasoning approaches
            \"\"\"
            prompt = GENERATECOT_PROMPT.format(input=input)
            tasks = [self._fill_node(GenerateCoTOp, prompt, mode="xml_fill") 
                    for _ in range(3)]
            results = await asyncio.gather(*tasks)
            return [self._parse_cot(r) for r in results]
"""
}

ScEnsemble = {
    "thought": {
        "Insights": "Solution consensus through multi-agent voting",
        "Overall Idea": "Comparative analysis of multiple solutions using LLM-as-judge",
        "Implementation": "1. Solution ranking mechanism\n2. Answer mapping system\n3. Majority voting"
    },
    "description": "Ensemble selector that analyzes multiple solution variants and returns the most consensus solution through simulated voting",
    "code": """
    class ScEnsemble(Operator):
        def __init__(self, llm: LLM, name: str = "ScEnsemble"):
            super().__init__(llm, name)
            self.voting_history = []

        async def __call__(self, solutions: List[str], problem: str):
            \"\"\"Determine optimal solution from multiple candidates
            Args:
                solutions: List of alternative solutions
                problem: Original problem statement
            Returns:
                Dictionary with selected solution and voting metadata
            \"\"\"
            answer_mapping = {chr(65+i): s for i,s in enumerate(solutions)}
            solution_text = "\\n\\n".join([f"{k}: {v}" for k,v in answer_mapping.items()])
            
            prompt = SC_ENSEMBLE_PROMPT.format(
                question=problem, 
                solutions=solution_text
            )
            
            response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")
            selected = response.get("solution_letter", "").strip().upper()
            
            return {
                "selected_solution": answer_mapping.get(selected, ""),
                "vote_distribution": response.get("votes", {}),
                "rationale": response.get("comparison_analysis", "")
            }
"""
}

SelfRefine = {
    "thought": {
        "Insights": "Enables iterative self-improvement of solutions through reflection",
        "Overall Idea": "Automated solution refinement using error feedback loops",
        "Implementation": "1. Error analysis prompt template\n2. Solution regeneration workflow\n3. Version tracking"
    },
    "description": "Iterative solution refiner that analyzes execution feedback and produces improved solution variants",
    "code": """
    class SelfRefine(Operator):
        def __init__(self, llm: LLM, name: str = "SelfRefine"):
            super().__init__(llm, name)
            self.version_control = []

        async def __call__(self, problem, solution, mode: str = None):
            \"\"\"Refine solution through critical self-analysis
            Args:
                problem: Original problem statement
                solution: Current solution version
                mode: Refinement strategy (strict/balanced/creative)
            Returns:
                Dictionary containing refined solution and revision notes
            \"\"\"
            prompt = SELFREFINE_PROMPT.format(
                problem=problem, 
                solution=solution,
                history="\\n".join(self.version_control[-3:])
            )
            response = await self._fill_node(SelfRefineOp, prompt, mode="xml_fill")
            self.version_control.append(f"Version {len(self.version_control)+1}: {response['revision_summary']}")
            return {
                "refined_solution": response["improved_solution"],
                "revision_notes": response["critical_analysis"]
            }
"""
}

Test = {
    "thought": {
        "Insights": "Automated validation through test case execution with diagnostic feedback",
        "Overall Idea": "Solution verification system with failure-driven refinement",
        "Implementation": "1. Test case extraction\n2. Sandboxed execution\n3. Error pattern analysis"
    },
    "description": "Automated testing operator that executes solutions against test cases and generates failure-informed refinement prompts",
    "code": """
    class Test(Operator):
        def __init__(self, llm: LLM, name: str = "Test"):
            super().__init__(llm, name)
            self.test_history = []

        async def _execute_test(self, solution: str, entry_point: str) -> dict:
            \"\"\"Safe execution wrapper with detailed error capture\"\"\"
            try:
                test_cases = extract_test_cases_from_jsonl(entry_point)
                env = {"__TEST_MODE__": True}
                exec(solution, env)
                results = {"passed": 0, "failed": []}
                
                for case in test_cases:
                    test_code = test_case_2_test_function(solution, case, entry_point)
                    try:
                        exec(test_code, env)
                        results["passed"] += 1
                    except AssertionError as e:
                        results["failed"].append({
                            "test_case": case,
                            "error_type": "AssertionError",
                            "error_message": str(e),
                            "traceback": traceback.format_exc()
                        })
                    except Exception as e:
                        results["failed"].append({
                            "test_case": case,
                            "error_type": "RuntimeError",
                            "error_message": str(e),
                            "traceback": traceback.format_exc()
                        })
                return results
            except Exception as e:
                return {"system_error": str(e)}

        async def __call__(self, problem: str, solution: str, entry_point: str, test_loop: int = 3):
            \"\"\"Iterative test-refine loop with failure analysis
            Args:
                problem: Target problem description
                solution: Initial solution code
                entry_point: Test case identifier
                test_loop: Maximum refinement iterations
            Returns:
                Final solution status with debugging history
            \"\"\"
            for iteration in range(test_loop):
                test_result = await self._execute_test(solution, entry_point)
                self.test_history.append(test_result)
                
                if not test_result.get("failed"):
                    return {"status": "success", "solution": solution, "iterations": iteration+1}
                    
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    error_log=json.dumps(test_result["failed"][:3], indent=2)
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["refined_solution"]
            
            final_test = await self._execute_test(solution, entry_point)
            return {
                "status": "partial" if final_test.get("failed") else "success",
                "solution": solution,
                "test_history": self.test_history
            }"""
}

EarlyStop = {
    "thought": {
        "Insights": "Prevents unnecessary computation through progress monitoring",
        "Overall Idea": "Conditional termination based on convergence metrics",
        "Implementation": "1. Performance plateau detection\n2. Resource usage monitoring\n3. Validation metric tracking"
    },
    "description": "Conditional termination operator that halts processing when quality plateaus or resource limits approach",
    "code": """
    class EarlyStop(Operator):
        def __init__(self, llm: LLM, name: str = "EarlyStop"):
            super().__init__(llm, name)
            self.metric_window = []
            self.stagnation_threshold = 3

        async def __call__(self, metrics: dict, resource_usage: dict):
            \"\"\"Evaluate stopping conditions based on performance trends
            Args:
                metrics: Current iteration quality metrics
                resource_usage: System resource consumption data
            Returns:
                Boolean indicating whether to terminate process
            \"\"\"
            self.metric_window.append(metrics["score"])
            if len(self.metric_window) > self.stagnation_threshold:
                recent_gain = abs(np.mean(self.metric_window[-3:]) - np.mean(self.metric_window[-6:-3]))
                if recent_gain < 0.01:
                    return True
            
            if resource_usage.get("memory_percent", 0) > 90:
                return True
                
            if metrics.get("improvement_rate", 0) < 0.05:
                return True
                
            return False
"""
}

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

base = """
# Overview
You are an expert machine learning researcher specializing in designing agentic systems. Your objective is to create building blocks such as prompts and control flows within these systems to solve complex tasks. Specifically, you aim to design an optimal agent that performs exceptionally on the HumanEval benchmark. The HumanEval dataset evaluates code generation capabilities in AI systems, consisting of 164 hand-crafted Python programming problems. Each problem includes: - A function signature with a docstring describing the task - Test cases to verify functional correctness

# Example Question from HumanEval
{"task_id": "HumanEval/102", "prompt": "\ndef choose_num(x, y):\n    \"\"\"This function takes two positive numbers x and y and returns the\n    biggest even integer number that is in the range [x, y] inclusive. If \n    there's no such number, then the function should return -1.\n\n    For example:\n    choose_num(12, 15) = 14\n    choose_num(13, 12) = -1\n    \"\"\"\n", "entry_point": "choose_num", "canonical_solution": "    if x > y:\n        return -1\n    if y % 2 == 0:\n        return y\n    if x == y:\n        return -1\n    return y - 1\n", "test": "def check(candidate):\n\n    # Check some simple cases\n    assert candidate(12, 15) == 14\n    assert candidate(13, 12) == -1\n    assert candidate(33, 12354) == 12354\n    assert candidate(5234, 5233) == -1\n    assert candidate(6, 29) == 28\n    assert candidate(27, 10) == -1\n\n    # Check some edge cases that are easy to work out by hand.\n    assert candidate(7, 7) == -1\n    assert candidate(546, 546) == 546\n\n"}

# Operator code template:
class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()

class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class Generate(Operator):
    GENERATE_PROMPT = '''
You are tasked with solving the following Python programming problem. Generate a complete, syntactically correct Python function that strictly adheres to the given requirements.

Problem:
{input}

Follow these steps:
1. Analyze the problem requirements and identify edge cases
2. Design a solution that passes all implied test cases
3. Implement the function with clear variable names and comments

Ensure:
- The code directly implements the requested functionality
- All parameters and return types match the problem specification
- Exception handling for edge cases is included when necessary '''

    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, mode: str = None):
        prompt = self.GENERATE_PROMPT.format(input=input)
        response = await self._fill_node(GenerateOp, prompt, mode="xml_fill")
        return response

# Discovered architecture archive
Here is the archive of the discovered operator architectures:
[ARCHIVE]

# Output Instruction and Example:
The output should be a JSON object with the following structure.The first key should be (“thought”), and it should capture your thought process for designing the next operator. The second key (“description”) corresponds to the brief description of your next operator. Finally, the last key (“code”) corresponds to the exact operator and its prompt in Python code that you would like to try. You must write COMPLETE CODE in “code”: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

- thought: Captures your thought process for designing the next operator.
  - Reason about what the next interesting operator should be.
  - Describe your reasoning and the overall concept behind the operator design.
  - Detail the implementation steps.
- description: A brief description of your next operator.
- code: The exact operator and its prompt in Python code. Ensure the code is complete, reliable, and reusable.


Here is an example of the output format for the next operator: 
[operator_example]

You must strictly follow the exact input/output interface used above. Also, it could be helpful to set the LLM’s role and temperature to further control the LLMs response. DON'T try to use some function that doesn't exist. In __call__(), you need to specify the instruction, input information, the prompt and the required output fields class for operators to do their specific part of the architecture. 

# Your task 
You are highly proficient in prompting techniques and well-versed with agentic systems from academic literature. Your goal is to maximize performance metrics by proposing innovative and effective new operators.
Instructions:
1. Analyze the Discovered Operators: Carefully review the operators in the archive to identify strengths, weaknesses, and areas for improvement.
2. Draw Insights: Extract lessons and insights from existing operators to inform the design of the next operator.
3. Innovate: Think creatively to design an operator that addresses current limitations or explores new functionalities, drawing inspiration from related agent papers or other research areas.
4. Design the Operator: Propose the next operator's `thought`, `description`, and `code` following the specified format.
5. Ensure Completeness: The generated code must be complete, reliable, and reusable, fitting seamlessly into the existing architecture.

Execution Steps:
1. Generate Output: Produce the `thought`, `description`, and `code` fields for the new operator, ensuring adherence to the guidelines.
2. Validate Output: Ensure the generated JSON is correctly formatted and the code is syntactically and functionally correct.

THINK OUTSIDE THE BOX and leverage interdisciplinary insights to enhance the agentic system's capabilities.
"""

def get_prompt_humaneval(current_archive):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt .replace("[operator_example]", json.dumps(operator_example))

    return system_prompt, prompt

def get_init_archive_humaneval():
    return [Generate, GenerateCoT, MultiGenerateCoT, ScEnsemble, Test, SelfRefine, EarlyStop]
