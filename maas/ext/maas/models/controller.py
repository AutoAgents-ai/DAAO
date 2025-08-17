import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.special import gammaln
from typing import List, Optional
from maas.ext.maas.models.utils import SentenceEncoder, sample_operators

# [384]
sentence_encoder = SentenceEncoder()

class DifficultyVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

def difficulty_guided_vae_loss(z, mu, logvar, is_solved, difficulty_scalar=None, gamma=0.1):
    """
    difficulty_scalar: 当前 predicted difficulty（可来自 decoder(z) 或 z.norm()）
    gamma: 控制 difficulty 上下调整的幅度
    """
    if difficulty_scalar is None:
        difficulty_scalar = z.norm(dim=-1)

    # 调整目标难度：成功 → 稍微降低，失败 → 稍微提升
    delta = gamma * (1 - 2 * is_solved.float())  # 成功→-γ, 失败→+γ
    target_difficulty = (difficulty_scalar + delta).clamp(0.0, 1.0)  # 保证在合法范围内

    # MSE loss：引导 difficulty 向目标靠拢
    difficulty_loss = F.mse_loss(difficulty_scalar, target_difficulty, reduction='mean')

    # KL 散度项（弱约束）
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return difficulty_loss + 0.01 * kl_loss


class QueryDifficultyEstimatorVAE(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int = 384,latent_dim: int = 32):
        super().__init__()
        self.encoder = encoder
        self.vae = DifficultyVAE(input_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, query_embedding):
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        z, mu, logvar = self.vae(query_embedding)
        difficulty_scalar = self.decoder(z)
        # z:[1, latent_dim]
        # print("================difficulty_scalar")
        # print(difficulty_scalar)
        return z, difficulty_scalar.squeeze(), mu, logvar


# --- Modified OperatorSelector ---
class OperatorSelector(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, latent_dim: int = 32, device=None, is_first_layer: bool = False):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_first_layer = is_first_layer
        self.difficulty_embed = nn.Linear(latent_dim, input_dim)

        if is_first_layer:
            self.operator_encoder = nn.Linear(input_dim, hidden_dim)
        else:
            self.operator_encoder = nn.Linear(input_dim * 2, hidden_dim)

        self.query_encoder = nn.Linear(input_dim*2 , hidden_dim)

    def forward(self, query_embed, operators_embed, z_difficulty, prev_operators_embed=None):
        if query_embed.dim() == 1:
            # [1,384]
            query_embed = query_embed.unsqueeze(0)

        
        difficulty_embed = self.difficulty_embed(z_difficulty)  # [1, input_dim]
        # print('===================operator1')
        # print(query_embed.shape, difficulty_embed.shape)
        query_cat = torch.cat([query_embed, difficulty_embed], dim=1)  # [1,input_dim *2]
        query_proj = F.normalize(self.query_encoder(query_cat), p=2, dim=1)

        if prev_operators_embed is not None and not self.is_first_layer:
            prev_expanded = prev_operators_embed[0].unsqueeze(0).expand(operators_embed.size(0), -1)
            # print('===================operator2')
            # print(operators_embed.shape, prev_expanded.shape)
            concat_embed = torch.cat([operators_embed, prev_expanded], dim=1)
        else:
            concat_embed = operators_embed

        op_proj = F.normalize(self.operator_encoder(concat_embed), p=2, dim=1)
        scores = torch.matmul(query_proj, op_proj.T)
        probs = F.softmax(scores, dim=1)
        log_probs = F.log_softmax(scores, dim=1)

        # [1, num_operator]
        return log_probs, probs


# --- Modified LLMRouter ---
class LLMRouter(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, latent_dim: int = 32, temp: float = 1.0, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.query_encoder = nn.Linear(input_dim*2, hidden_dim)
        self.operator_encoder = nn.Linear(input_dim, hidden_dim)
        self.llm_encoder = nn.Linear(input_dim, hidden_dim)
        self.difficulty_embed = nn.Linear(latent_dim, input_dim)
        self.combined_encoder = nn.Linear(hidden_dim * 2, hidden_dim)

        self.temp = temp

    def forward(self, query_embed, z_difficulty, selected_operator_embeddings, llms, prev_llm_embed=None):
        if query_embed.dim() == 1:
            # [1,384]
            query_embed = query_embed.unsqueeze(0)
        N_op = selected_operator_embeddings.size(0)
        N_llm = llms.size(0)

        # Embed query + difficulty
        difficulty_embed = self.difficulty_embed(z_difficulty)
        # print('===================llmrouter1')
        # print(query_embed.shape, difficulty_embed.shape)
        query_cat = torch.cat([query_embed, difficulty_embed], dim=1)
        # [1, hidden_dim]
        query_proj = F.normalize(self.query_encoder(query_cat), p=2, dim=1)

        # Embed all LLMs
        # [n, hidden_dim]
        llm_proj = F.normalize(self.llm_encoder(llms), p=2, dim=1)

        selected_llm_indices = []
        log_probs = []
        all_probs = []

        for i in range(N_op):
            op_embed = selected_operator_embeddings[i].unsqueeze(0)

            op_proj = F.normalize(self.operator_encoder(op_embed), p=2, dim=1)

            # Combine query + operator
            # print('===================llmrouter2')
            # print(query_proj.shape, op_proj.shape)
            query_op = torch.cat([query_proj, op_proj], dim=1)  # shape: [1, 2 * hidden_dim]

            # Optionally map it back to hidden_dim
            combined_proj = F.normalize(self.combined_encoder(query_op), p=2, dim=1)  # shape: [1, hidden_dim]

            # Compute similarity
            # [1, llm_sum]
            score = torch.matmul(combined_proj, llm_proj.T)
            probs = F.softmax(score, dim=1)

            selected_idx = torch.multinomial(probs, num_samples=1).item()
            selected_llm_indices.append(selected_idx)

            log_prob = F.log_softmax(score, dim=1)
            # 修改返回的 log_probs 为 selected 的 log_prob
            log_probs.append(log_prob[0, selected_idx])  # ✅ 改成这个

            all_probs.append(probs)

        # # 转为张量
        # log_probs = torch.stack(log_probs).unsqueeze(1)      # shape: [N_op, 1]
        # all_probs = torch.stack(all_probs)                    # shape: [N_op, N_llm]

        return selected_llm_indices, torch.stack(log_probs), all_probs



class MultiLayerController(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 32, num_layers: int = 4, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layers = nn.ModuleList([
            OperatorSelector(input_dim,hidden_dim=hidden_dim, device=self.device, is_first_layer=(i == 0))
            for i in range(num_layers)
        ])
        self.num_layers = num_layers

        self.sentence_encoder = sentence_encoder
        self.difficulty_estimator = QueryDifficultyEstimatorVAE(self.sentence_encoder, input_dim, hidden_dim)
        self.llm_router = LLMRouter(input_dim=input_dim, device=self.device)

    def forward(self, query, operators_embedding, llms_embedding, operator_names):
        query_embedding = self.sentence_encoder(query).to(self.device)
        # 【num_op, 384】
        operators_embedding = operators_embedding.to(self.device)
        llms_embedding = llms_embedding.to(self.device)

        # [latent_dim]
        # print("==============z_difficulty")
        # z[1,latent_dim]
        z_difficulty, difficulty_scalar, mu, logvar = self.difficulty_estimator(query_embedding)   
        print(z_difficulty.shape)
        max_exec_layers = int((difficulty_scalar * self.num_layers).clamp(1, self.num_layers).item())

        log_probs_layers = []
        selected_operators_layers = []
        selected_llms_layers = []
        prev_operators = None
        prev_llm_embed = None

        for layer_idx, layer in enumerate(self.layers[:max_exec_layers]):
            if layer_idx == 0:
                log_probs, probs = layer(query_embedding, operators_embedding, z_difficulty)
            else:
                log_probs, probs = layer(query_embedding, operators_embedding, z_difficulty, prev_operators)

            # [operator_nums]
            probs_1d = probs.squeeze(0)
            log_probs_1d = log_probs.squeeze(0)
            # 返回索引
            selected_indices = sample_operators(probs_1d, threshold=0.3)

            if selected_indices.numel() == 0:
                selected_operators_layers.append([])
                selected_llms_layers.append([])
                log_probs_layers.append(torch.tensor(0.0, device=self.device))
                continue

            selected_indices = selected_indices.to(operators_embedding.device)
            selected_names = [operator_names[idx] for idx in selected_indices.cpu().tolist()]
            

            if layer_idx == 0:
                if not any("generate" in name.lower() for name in selected_names):
                    try:
                        generate_idx = selected_names.index("Generate")
                    except ValueError:
                        generate_idx = 0
                    selected_indices = torch.tensor([generate_idx], device=self.device)
                    selected_names = ["Generate"]
                elif "generate" not in selected_names[0].lower() and any("generate" in name.lower() for name in selected_names):
                    for idx, name in enumerate(selected_names):
                        if "generate" in name.lower():
                            selected_names = [selected_names[idx]] + selected_names[:idx] + selected_names[idx+1:]
                            try:
                                new_first_idx = selected_names.index(selected_names[0])
                            except ValueError:
                                new_first_idx = 0
                            new_indices = [new_first_idx] + [selected_names.index(n) for n in selected_names[1:]]
                            selected_indices = torch.tensor(new_indices, device=self.device)
                            break

            selected_operator_embeddings = operators_embedding[selected_indices]

            # LLM routing：每个 operator 对应一个 LLM
            llm_indices, llm_log_probs, _ = self.llm_router(
                query_embed=query_embedding,
                z_difficulty=z_difficulty,
                selected_operator_embeddings=selected_operator_embeddings,
                llms=llms_embedding,
            )

            # 选中的 operator 的 log prob
            op_log_probs = log_probs_1d[selected_indices]         # [N_op]
            flat_llm_log_prob = llm_log_probs.sum()  # ✅ 把每个选中 LLM 的 log prob 加总
            layer_log_prob = op_log_probs.sum() + flat_llm_log_prob

            log_probs_layers.append(layer_log_prob)
            selected_operators_layers.append(selected_names)
            selected_llms_layers.append(llm_indices)

            prev_operators = selected_operator_embeddings

        return log_probs_layers, selected_operators_layers, selected_llms_layers, z_difficulty, difficulty_scalar, mu, logvar