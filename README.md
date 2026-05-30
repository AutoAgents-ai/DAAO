# DAAO

**DAAO** 是一个**"可学习的多智能体工作流架构搜索"（Multi-agent Architecture Search）**框架，基于  MaAS 改装而来。

核心思路：训练的不是某个大模型，而是训练一个"调度大脑"——学会针对题目难度，动态决定用哪些推理算子、配哪个 LLM、走几层，用强化学习把"答得准又省钱"的编排策略学出来。

## 🙏 Acknowledgement

Special thanks to the following repositories for their invaluable code and prompt.

Our prompt is partially adapted from [ADAS](https://github.com/ShengranHu/ADAS), [AgentSquare](https://github.com/tsinghua-fib-lab/AgentSquare/tree/main), and [AFLOW](https://github.com/geekan/MetaGPT/tree/main/examples/aflow). Our code and operators are partially adapted from [AFLOW](https://github.com/geekan/MetaGPT/tree/main/examples/aflow).

## 快速开始

### 1. 环境准备

```bash
pip install -e .
```

### 2. 配置模型

DAAO 从两个路径加载配置并合并（用户级覆盖项目级）：

1. **项目级**：`<项目根目录>/config/config2.yaml`
2. **用户级**：`~/.maas/config2.yaml`

> 推荐在 `~/.maas/config2.yaml` 中配置，这样多个项目可共享同一份密钥。也可直接编辑项目根目录下的 `config/config2.yaml`。

编辑配置文件，填入可用的 LLM 模型：

```yaml
models:
  gpt-4o-mini:
    api_type: "openai"
    model: "gpt-4o-mini"
    base_url: ""   # OpenAI 官方留空，代理/转发填对应地址
    api_key: ""    # 你的 API Key
  qwen2.5:7b:
    api_type: "ollama"
    model: "qwen2.5:7b"
    base_url: ""
    api_key: ""
  llama3.1:70b:
    api_type: "ollama"
    model: "llama3.1:70b"
    base_url: ""
    api_key: ""

# 可供 LLMRouter 选择的模型列表（不配置则默认使用所有 models）
available_llms:
  - "gpt-4o-mini"
  - "qwen2.5:7b"
  - "llama3.1:70b"
```

> **说明**：`models` 定义所有可用模型的连接信息；`available_llms` 指定 LLMRouter 可路由的模型子集。若不配置 `available_llms`，Router 默认使用 `models` 中的全部模型。模型名称必须在 `models` 中有对应条目。

### 3. 运行

#### 训练（学习控制器）

```bash
python examples/maas/optimize.py --dataset GSM8K \
    --opt_model_name gpt-4o-mini --exec_model_name gpt-4o-mini \
    --sample 4 --round 1 --batch_size 4 --lr 0.01
```

参数说明：
- `--dataset`：数据集，可选 `GSM8K` / `MATH` / `HumanEval`
- `--opt_model_name`：用于优化决策的 LLM 名称，必须在 `models` 中已配置（建议用便宜的模型）
- `--exec_model_name`：用于执行推理的主 LLM 名称，必须在 `models` 中已配置
- `--sample`：每轮采样次数
- `--round`：当前训练轮次编号（决定结果保存到哪个 round 目录）
- `--batch_size`：训练批次大小
- `--lr`：学习率
- `--is_textgrad`：是否启用 TextGrad 文本梯度优化（默认关闭）

#### 测试（评估已训练的控制器）

```bash
python examples/maas/optimize.py --dataset GSM8K \
    --opt_model_name gpt-4o-mini --exec_model_name gpt-4o-mini \
    --is_test --round 1
```

> **注意**：测试时 `--round` 指定加载第几轮训练产出的 checkpoint（对应训练时的 `--round` 值）。`--opt_model_name` 和 `--exec_model_name` 仍需指定以初始化 LLM。

## 核心架构

```
optimize.py (命令行入口)
   └─ Optimizer                              # 构建控制器 + Adam 优化器
       └─ EvaluationUtils                    # 编排评估流程
           └─ Evaluator                      # 按数据集创建 Benchmark + 配置 Workflow 图
               └─ Benchmark (GSM8K/MATH/HumanEval)  # 训练循环 + 算 reward + 回传梯度
                   └─ Workflow (graph.py)            # 真正执行一道题
                       ├─ MultiLayerController       # 神经网络：做架构决策
                       └─ Operators + 多个 LLM       # 按决策真去调大模型解题
```

## 关键模块

| 模块 | 路径 | 作用 |
|------|------|------|
| `MultiLayerController` | `daao/ext/maas/models/controller.py` | 神经网络控制器，包含难度估计 VAE + 算子选择 + LLM 路由 |
| `LLMRouter` | `daao/ext/maas/models/controller.py` | 根据题目+算子，从 `available_llms` 中选择合适的 LLM |
| `Optimizer` | `daao/ext/maas/scripts/optimizer.py` | 主控类：训练循环编排、测试入口 |
| `Evaluator` | `daao/ext/maas/scripts/evaluator.py` | 按数据集创建 Benchmark 实例、配置 Workflow 图 |
| `Benchmark` | `daao/ext/maas/benchmark/benchmark.py` | 训练循环：跑题 → 计算 reward → 策略梯度更新 |
| `Workflow` | `daao/ext/maas/scripts/optimized/*/train/graph.py` | 执行图：根据控制器决策，逐层调用算子和 LLM |
| `EvaluationUtils` | `daao/ext/maas/scripts/optimizer_utils/evaluation_utils.py` | 评估流程编排，连接 Optimizer 与 Evaluator |
