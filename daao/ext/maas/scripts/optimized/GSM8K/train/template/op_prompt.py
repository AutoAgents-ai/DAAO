SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""

SELFREFINE_PROMPT = """
You are an assistant specialized in refining solutions to problems.

Problem:
{problem}

Solution:
{solution}

Instruction:
Analyze the above solution for any errors or suboptimal aspects. Make iterative improvements to enhance its correctness and efficiency. Provide the refined solution below.
"""

GENERATE_COT_PROMPT = """
Mathematical Reasoning Instruction
{instruction}

Current Problem:
{input}

Demonstration Examples (GSM8K/MATH style):

1. Problem: Find all real solutions to $x^4 - 3x^2 - 4 = 0$
   Analysis: 
   Let $y = x^2$, transforming to quadratic equation:  
   $y^2 - 3y - 4 = 0$  
   Factorization: $(y-4)(y+1)=0$ → $y=4$ or $y=-1$  
   Since $y=x^2 \geq 0$, discard $y=-1$  
   Solve $x^2=4$ → $x=\pm 2$  
   \boxed{-2}, \boxed{2}

2. Problem: Cone with height 12cm, radius 6cm. Find volume.
   Analysis:
   Volume formula: $V = \frac{1}{3}\pi r^2 h$  
   Substitute values: $V = \frac{1}{3}\pi(6)^2(12)$  
   Calculate: $6^2=36$ → $36\times12=432$  
   Final volume: $\frac{432}{3}\pi = 144\pi$  
   \boxed{144\pi}


Solution Protocol:
1. Parse problem statement carefully
2. Identify relevant mathematical concepts
3. Perform stepwise symbolic derivation
4. Verify intermediate results
5. Present final answer in boxed notation

"""
