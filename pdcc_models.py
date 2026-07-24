import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F


# 这是一个Top-k路由网络的实现，常用于混合专家模型(MoE)中
class TopkRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k):
# embed_dim：输入特征的维度
# num_experts：专家（Expert）的数量
# top_k：每个样本选择的前k个专家
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(embed_dim, num_experts)

# self.linear：一个线性层，将输入从 embed_dim 维映射到 num_experts 维

    def forward(self, x):
# x 的形状 (batch_size, embed_dim)
        logits = self.linear(x)
# logits 的形状为(batch_size, num_experts)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
# 作用：找出每个样本得分最高的k个专家
# self.top_k：要选择的专家数量（比如2）
# dim=-1：在最后一个维度（专家维度）上操作
# logits.topk 是 PyTorch 张量的一个方法，用于获取张量中前k个最大的值及其索引
        zeros = torch.full_like(logits, float('-inf'))
# 创建一个与logits形状相同的矩阵，所有元素都是负无穷
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
# -1：在最后一个维度操作
# output = tensor.scatter(dim, index, src)
# dim：要在哪个维度上进行散射操作
# index：索引张量，指定要更新的位置
# src：源张量，提供要填充的值
        router_output = F.softmax(sparse_logits, dim=-1)
# F.softmax 是一个激活函数，它将任意实数值转换为概率分布。
        return router_output, indices


# 定义一个文本模型
class TextModel(nn.Module):
# 多模态文本模型的初始化方法
    def __init__(self, hyp_params):
# hyp_params：包含所有超参数的对象
# 调用父类 nn.Module 的构造函数
        super(TextModel, self).__init__()
        self.orig_l_len = hyp_params.l_len          # 原始语言序列长度x
        self.orig_d_l = hyp_params.orig_d_l         # 原始语言特征维度
        self.embed_dim = hyp_params.embed_dim       # 嵌入维度
        self.out_dropout = hyp_params.out_dropout   # 输出dropout率
        self.language = hyp_params.language         # 语言类型
        self.finetune = hyp_params.finetune         # 是否微调

        self.output_dim = hyp_params.output_dim     # 输出维度                # This is actually not a hyperparameter :-)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=1, bias=False)
# 1D卷积投影层
# 作用：将原始语言特征维度投影到统一的嵌入维度
# self.orig_d_l：输入通道数（原始特征维度）
# self.embed_dim：输出通道数（目标嵌入维度）
# kernel_size=1：1x1卷积，相当于线性变换
# bias=False：不使用偏置
# 效果：(batch, orig_d_l, seq_len) → (batch, embed_dim, seq_len)

        # Prepare BERT model
        self.text_model = BertTextEncoder(language=hyp_params.language)
# 作用：使用预训练的BERT模型作为文本编码器

        # Unimodal encoder
        self.encoder_l = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)
# Transformer编码器
# 作用：在BERT特征之上进一步编码，捕捉序列依赖关系
# d_model=self.embed_dim：输入维度
# nhead=hyp_params.nhead：注意力头数
# dim_feedforward=4 * self.embed_dim：前馈网络维度（通常是4倍）
# norm_first=True：先进行层归一化（更稳定）
# num_layers=hyp_params.transformer_layers：Transformer层数

# 作用：将Transformer输出映射到目标维度
        # Projection layers
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, self.output_dim)
# proj1 和 proj2：中间投影层，增加模型表达能力
# out_layer：最终输出层，映射到分类/回归维度

    def forward(self, x_l):
        x_l = self.text_model(x_l, use_finetune=self.finetune)
# 作用：使用BERT模型提取文本特征
# x_l：原始文本输入（tokenIDs）
# use_finetune = self.finetune：控制是否微调BERT参数
        #################################################################################
        # Project the textual features
        x_l = x_l.transpose(1, 2)  # (bs, seq, embed) → (bs, embed, seq)
        proj_x_l = self.proj_l(x_l)  # (bs, embed, seq)
        proj_x_l = proj_x_l.permute(2, 0, 1)  # (seq, bs, embed)
# transpose(1, 2)：为1D卷积准备，将特征维度放到第1维
# self.proj_l(x_l)：1x1卷积，统一特征维度
# permute(2, 0, 1)：调整为Transformer期望的输入格式(seq_len, batch_size, embed_dim)
        #################################################################################
        # Unimodal encoder
        h_l = self.encoder_l(proj_x_l)
# 作用：使用Transformer进一步编码文本特征
# 输入：(seq_len, batch_size, embed_dim)
# 输出：(seq_len, batch_size, embed_dim)（Transformer编码后的序列）
        #################################################################################
        # Predict
        h_l = torch.mean(h_l, dim=0)  # (bs, embed)
        h_l = F.normalize(h_l, p=2, dim=-1)
        last_hs = h_l
# torch.mean(h_l, dim=0)：在序列维度上取平均，将变长序列转换为固定长度表示
# F.normalize(h_l, p=2, dim=-1)：L2归一化，使特征向量模长为1，2表示L2范数
# last_hs：保存归一化后的特征
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
# p 表示丢弃概率 self.training 表示是否启用训练模式
        last_hs_proj += last_hs
# 残差连接
        final_output = self.out_layer(last_hs_proj)
# 最终输出层
        outputs = {
            'pred': final_output,           # 主要预测结果
            'last_hs_proj': last_hs_proj,   # 投影后的特征（含残差）
            'h_l': h_l                      # 原始编码特征
        }

        return outputs


# 定义一个音频模型
class AudioModel(nn.Module):
# 音频模型初始化方法
    def __init__(self, hyp_params):
        super(AudioModel, self).__init__()
        self.orig_a_len = hyp_params.a_len          # 原始音频序列长度
        self.orig_d_a = hyp_params.orig_d_a         # 原始音频特征维度
        self.embed_dim = hyp_params.embed_dim       # 嵌入维度
        self.out_dropout = hyp_params.out_dropout   # 输出dropout率

        self.output_dim = hyp_params.output_dim     # 输出维度          # This is actually not a hyperparameter :-)
        self.norm_a = nn.BatchNorm1d(self.orig_d_a) # 批归一化
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=1, bias=False)

        # Unimodal encoder
        self.encoder_a = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)
# Transformer编码器
# 作用：捕捉序列依赖关系
# d_model=self.embed_dim：输入维度
# nhead=hyp_params.nhead：注意力头数
# dim_feedforward=4 * self.embed_dim：前馈网络维度（通常是4倍）
# norm_first=True：先进行层归一化（更稳定）
# num_layers=hyp_params.transformer_layers：Transformer层数

# 作用：将Transformer输出映射到目标维度
        # Projection layers
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, x_a):
        #################################################################################
        # Project the audio features
        x_a = x_a.transpose(1, 2)    # (bs, seq, embed) → (bs, embed, seq)
        x_a = self.norm_a(x_a)       # 批归一化
        proj_x_a = self.proj_a(x_a)  # (bs, embed, seq)
        proj_x_a = proj_x_a.permute(2, 0, 1)  # (seq, bs, embed)
        #################################################################################
        # Unimodal encoder
        h_a = self.encoder_a(proj_x_a)
# 作用：使用Transformer进一步编码音频特征
# 输入：(seq_len, batch_size, embed_dim)
# 输出：(seq_len, batch_size, embed_dim)（Transformer编码后的序列）
        #################################################################################
        # Predict
        h_a = torch.mean(h_a, dim=0)  # (bs, embed)
# torch.mean(h_l, dim=0)：在序列维度上取平均，将变长序列转换为固定长度表示
        h_a = F.normalize(h_a, p=2, dim=-1)
        last_hs = h_a
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        final_output = self.out_layer(last_hs_proj)
        outputs = {
            'pred': final_output,
            'last_hs_proj': last_hs_proj,
            'h_a': h_a
        }

        return outputs


# 定义一个视觉模型
class VisionModel(nn.Module):
# 视觉模型初始化方法
    def __init__(self, hyp_params):
        super(VisionModel, self).__init__()
        self.orig_v_len = hyp_params.v_len          # 原始视觉序列长度
        self.orig_d_v = hyp_params.orig_d_v         # 原始视觉特征维度
        self.embed_dim = hyp_params.embed_dim       # 嵌入维度
        self.out_dropout = hyp_params.out_dropout   # 输出dropout率

        self.output_dim = hyp_params.output_dim     # 输出维度          # This is actually not a hyperparameter :-)
        self.norm_v = nn.BatchNorm1d(self.orig_d_v) # 批归一化
        self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=1, bias=False)   # 卷积

        # Unimodal encoder
        self.encoder_v = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                                          nhead=hyp_params.nhead,
                                                                          dim_feedforward=4 * self.embed_dim,
                                                                          norm_first=True),
                                               num_layers=hyp_params.transformer_layers)
# Transformer编码器
# 作用：捕捉序列依赖关系
# d_model=self.embed_dim：输入维度
# nhead=hyp_params.nhead：注意力头数
# dim_feedforward=4 * self.embed_dim：前馈网络维度（通常是4倍）
# norm_first=True：先进行层归一化（更稳定）
# num_layers=hyp_params.transformer_layers：Transformer层数

        # Projection layers
        self.proj1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_layer = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, x_v):
        #################################################################################
        # Project the visual features
        x_v = x_v.transpose(1, 2)    # (bs, seq, embed) → (bs, embed, seq)
        x_v = self.norm_v(x_v)       # 批归一化
        proj_x_v = self.proj_v(x_v)  # (bs, embed, seq)
        proj_x_v = proj_x_v.permute(2, 0, 1)  # (seq, bs, embed)
        #################################################################################
        # Unimodal encoder
        h_v = self.encoder_v(proj_x_v)
        #################################################################################
        # Predict
        h_v = torch.mean(h_v, dim=0)  # (bs, embed)
# torch.mean(h_l, dim=0)：在序列维度上取平均，将变长序列转换为固定长度表示
        h_v = F.normalize(h_v, p=2, dim=-1)
        last_hs = h_v
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        final_output = self.out_layer(last_hs_proj)
        outputs = {
            'pred': final_output,
            'last_hs_proj': last_hs_proj,
            'h_v': h_v
        }

        return outputs


class PDCCModel(nn.Module):
    def __init__(self, hyp_params):
        super(PDCCModel, self).__init__()
        self.orig_l_len, self.orig_a_len, self.orig_v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.embed_dim = hyp_params.embed_dim
        self.top_k = hyp_params.top_k
        self.out_dropout = hyp_params.out_dropout
        self.pcrp_temperature = hyp_params.pcrp_temperature
        self.language = hyp_params.language
        self.finetune = hyp_params.finetune

        self.output_dim = hyp_params.output_dim

        # ===== PDCC-MER module switches =====
        self.use_pcrp = getattr(hyp_params, "use_pcrp", False)
        self.pcrp_steps = getattr(hyp_params, "pcrp_steps", 1)
        self.pcrp_strength = getattr(hyp_params, "pcrp_strength", 0.5)

        self.use_rccr = getattr(hyp_params, "use_rccr", False)
        self.rccr_tau = getattr(hyp_params, "rccr_tau", 1.0)
        self.rccr_lambda = getattr(hyp_params, "rccr_lambda", 0.1)

        self.norm_a = nn.BatchNorm1d(self.orig_d_a)
        self.norm_v = nn.BatchNorm1d(self.orig_d_v)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.embed_dim, kernel_size=1, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.embed_dim, kernel_size=1, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.embed_dim, kernel_size=1, bias=False)

        self.text_model = BertTextEncoder(language=hyp_params.language)

        self.encoder_l = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=hyp_params.nhead,
                dim_feedforward=4 * self.embed_dim,
                norm_first=True
            ),
            num_layers=hyp_params.transformer_layers
        )
        self.encoder_a = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=hyp_params.nhead,
                dim_feedforward=4 * self.embed_dim,
                norm_first=True
            ),
            num_layers=hyp_params.transformer_layers
        )
        self.encoder_v = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=hyp_params.nhead,
                dim_feedforward=4 * self.embed_dim,
                norm_first=True
            ),
            num_layers=hyp_params.transformer_layers
        )

        self.router = TopkRouter(embed_dim=self.embed_dim, num_experts=3, top_k=self.top_k)

        self.proj1s = nn.ModuleList([])
        self.proj2s = nn.ModuleList([])
        self.out_layers = nn.ModuleList([])

        for _ in range(3):
            self.proj1s.append(nn.Linear(self.embed_dim, self.embed_dim))
            self.proj2s.append(nn.Linear(self.embed_dim, self.embed_dim))
            self.out_layers.append(nn.Linear(self.embed_dim, self.output_dim))

    def forward(self, x_l, x_a, x_v):
        x_l = self.text_model(x_l, use_finetune=self.finetune)
        #################################################################################
        # Project the textual/audio/visual features
        x_l = x_l.transpose(1, 2)   # 原来的维度是(bs, seq, embed) 转换为(bs, embed, seq)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        x_a = self.norm_a(x_a)
        x_v = self.norm_v(x_v)
        proj_x_l = self.proj_l(x_l)
        proj_x_a = self.proj_a(x_a)
        proj_x_v = self.proj_v(x_v)  # (bs, embed, seq)
        proj_x_l = proj_x_l.permute(2, 0, 1)  # (seq, bs, embed)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        #################################################################################
        # Unimodal encoder
        h_l = self.encoder_l(proj_x_l)
        h_a = self.encoder_a(proj_x_a)
        h_v = self.encoder_v(proj_x_v)
        h_l = torch.mean(h_l, dim=0)  # (bs, embed)
        h_a = torch.mean(h_a, dim=0)  # (bs, embed)
        h_v = torch.mean(h_v, dim=0)  # (bs, embed)
        h_l = F.normalize(h_l, p=2, dim=-1)
        h_a = F.normalize(h_a, p=2, dim=-1)
        h_v = F.normalize(h_v, p=2, dim=-1)
        #################################################################################
# 多模态特征融合准备阶段
        # Multimodal fusion
        h_ll = h_l.unsqueeze(1)  # (bs, 1, embed)
        h_aa = h_a.unsqueeze(1)
        h_vv = h_v.unsqueeze(1)
        x = torch.cat((h_ll, h_aa, h_vv), dim=1)  # (bs, 3, embed)
        # 作用：在维度1（模态维度）上拼接三个模态的特征

        # ===== DEBUG: optionally perturb PFM input x for robustness check =====
        if getattr(self, "debug_pfm", False):
            # 保存未扰动的 PFM 输入
            self._pfm_x_clean = x.detach().cpu()

            k = float(getattr(self, "pfm_noise_k", 0.0))
            if k > 0:
                g = torch.Generator(device=x.device)
                g.manual_seed(int(getattr(self, "pfm_noise_seed", 0)))

                std = x.std().detach() + 1e-6
                seed = int(getattr(self, "pfm_noise_seed", 0))

                cpu_state = torch.get_rng_state()
                cuda_states = None
                if x.is_cuda:
                    cuda_states = torch.cuda.get_rng_state_all()

                torch.manual_seed(seed)
                if x.is_cuda:
                    torch.cuda.manual_seed_all(seed)

                x = x + k * std * torch.randn_like(x)

                # 恢复 RNG 状态
                torch.set_rng_state(cpu_state)
                if x.is_cuda and cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)

            # 保存扰动后的 PFM 输入（方便你确认确实变了）
            self._pfm_x_used = x.detach().cpu()

        # ===== PCRP =====
        cosine_similarity = torch.bmm(x, x.transpose(1, 2))
        sim_scores = F.softmax(cosine_similarity / self.pcrp_temperature, dim=-1)

        if getattr(self, "debug_pfm", False):
            self._pfm_pre = sim_scores.detach().cpu()

        if self.use_pcrp:
            S = sim_scores.clone()
            steps = max(1, int(self.pcrp_steps))
            for _ in range(steps - 1):
                S = torch.bmm(S, S)
                S = F.softmax(S / self.pcrp_temperature, dim=-1)

            sim_scores = (1.0 - self.pcrp_strength) * sim_scores + self.pcrp_strength * S
            sim_scores = sim_scores / (sim_scores.sum(dim=-1, keepdim=True) + 1e-12)

        if getattr(self, "debug_pfm", False):
            self._pfm_post = sim_scores.detach().cpu()

        x = torch.bmm(sim_scores, x)

        # ===== RCCR: reliability prior by predictive entropy =====
        expert_logits = []
        for i in range(3):
            h = x[:, i, :]
            h_proj = self.proj2s[i](
                F.dropout(F.relu(self.proj1s[i](h)), p=self.out_dropout, training=self.training)
            )
            h_proj = h_proj + h
            z_i = self.out_layers[i](h_proj)
            expert_logits.append(z_i)

        z_t, z_a, z_v = expert_logits

        if self.use_rccr:
            p_t = F.softmax(z_t, dim=-1)
            p_a = F.softmax(z_a, dim=-1)
            p_v = F.softmax(z_v, dim=-1)

            def entropy(p, eps=1e-12):
                return -(p * (p + eps).log()).sum(dim=-1)

            reliability_uncertainty = torch.stack(
                [entropy(p_t), entropy(p_a), entropy(p_v)], dim=-1
            )
            reliability_prior = F.softmax(-reliability_uncertainty / self.rccr_tau, dim=-1)

            fused_x = torch.sum(x * reliability_prior.unsqueeze(-1), dim=1)
        else:
            reliability_prior = None
            fused_x = torch.sum(x, dim=1)

        # ===== DEBUG: (optional) inject controlled perturbation on router input fused_x =====
        if getattr(self, "debug_mcr", False):
            self._mcr_fused_clean = fused_x.detach().cpu()

            k = float(getattr(self, "mcr_noise_k", 0.0))
            if k > 0:
                seed = int(getattr(self, "mcr_noise_seed", 0))

                cpu_state = torch.get_rng_state()
                cuda_state = None
                if fused_x.is_cuda:
                    cuda_state = torch.cuda.get_rng_state_all()

                torch.manual_seed(seed)
                if fused_x.is_cuda:
                    torch.cuda.manual_seed_all(seed)

                std = fused_x.std().detach() + 1e-6
                fused_x = fused_x + k * std * torch.randn_like(fused_x)

                torch.set_rng_state(cpu_state)
                if fused_x.is_cuda and cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

            self._mcr_fused_used = fused_x.detach().cpu()

        # 3) Top-k router 正常工作
        gating_output, indices = self.router(fused_x)
        gate_raw = gating_output.clone()

        if getattr(self, "debug_mcr", False):
            if reliability_prior is not None:
                self._mcr_dom = reliability_prior.detach().cpu()
            self._mcr_gate_raw = gating_output.detach().cpu()

        if self.use_rccr and reliability_prior is not None:
            gating_output = (1.0 - self.rccr_lambda) * gating_output + self.rccr_lambda * reliability_prior
            gating_output = gating_output / (gating_output.sum(dim=-1, keepdim=True) + 1e-12)

        if getattr(self, "debug_mcr", False):
            self._mcr_gate_cal = gating_output.detach().cpu()

        bs = x.shape[0]

        # 1) 三个专家对所有样本都计算 logits（pred_t/a/v 全量）
        expert_logits = []
        for i in range(3):  # i=0: text, i=1: audio, i=2: vision
            last_hs = x[:, i, :]  # (bs, embed_dim)

            last_hs_proj = self.proj2s[i](
                F.dropout(
                    F.relu(self.proj1s[i](last_hs)),
                    p=self.out_dropout,
                    training=self.training
                )
            )
            last_hs_proj = last_hs_proj + last_hs  # residual

            out_i = self.out_layers[i](last_hs_proj)  # (bs, output_dim)
            expert_logits.append(out_i)

        pred_t, pred_a, pred_v = expert_logits[0], expert_logits[1], expert_logits[2]

        # 2) 用（已校准的）gating_output 对三个专家 logits 加权融合得到 pred
        # gating_output: (bs, 3)
        stacked = torch.stack(expert_logits, dim=1)  # (bs, 3, output_dim)
        final_output = torch.sum(gating_output.unsqueeze(-1) * stacked, dim=1)  # (bs, output_dim)

        outputs = {
            'pred': final_output,
            'pred_t': pred_t,
            'pred_a': pred_a,
            'pred_v': pred_v,

            'h_l': h_l,
            'h_a': h_a,
            'h_v': h_v,

            'h_pcrp': x.mean(dim=1),
            'h_fused': fused_x,

            'reliability_prior': reliability_prior,
            'gate_raw': gate_raw,
            'gate': gating_output,
            'topk_idx': indices,
        }
        return outputs


# 用于在训练过程中维护梯度或参数的指数移动平均
# EMA_t = momentum × EMA_{t-1} + (1 - momentum) × current_value
class EMA:
    def __init__(self, momentum=0.999):     # 动量参数，控制历史信息的保留程度
        self.momentum = momentum
        self.global_grad = None             # 存储移动平均值的变量

    def update(self, cur_global_grad):
        if self.global_grad is None:
            self.global_grad = cur_global_grad
        else:
            self.global_grad = self.momentum * self.global_grad + (1 - self.momentum) * cur_global_grad

    def step(self, new_momentum):
        self.new_momentum = new_momentum
# 作用：更新动量参数（可能在训练过程中调整）


class BertTextEncoder(nn.Module):
    def __init__(self, language='en'):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('/data/Lab105/zhouxin/MSA_Datasets/bert_en',
                                                             do_lower_case=True)
            self.model = model_class.from_pretrained('/data/Lab105/zhouxin/MSA_Datasets/bert_en')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('/data/Lab105/zhouxin/MSA_Datasets/bert_cn')
            self.model = model_class.from_pretrained('/data/Lab105/zhouxin/MSA_Datasets/bert_cn')

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, text, use_finetune):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, features, labels):
        """
        features: [num_samples, feature_dim]
        labels:   [num_samples] or [num_samples, 1]
        """
        # 1) 特征归一化，避免相似度无限大
        features = F.normalize(features, dim=-1)  # 需要确保已导入 torch.nn.functional as F

        # 2) 相似度矩阵 / 温度
        sim = torch.matmul(features, features.T)  # [N, N]
        sim = sim / self.temperature

        # 3) 数值稳定：每行减去最大值，防止 exp 溢出
        sim = sim - sim.max(dim=1, keepdim=True)[0]  # [N, N]

        # 4) 掩码
        labels = labels.view(-1, 1).contiguous()     # [N, 1]
        mask_positive = torch.eq(labels, labels.T).float()  # [N, N]
        mask_positive.fill_diagonal_(0)

        exp_sim = torch.exp(sim)  # 不再爆炸成 inf

        numerator = torch.sum(exp_sim * mask_positive, dim=1)  # [N]
        denominator = torch.sum(exp_sim, dim=1) - exp_sim.diag()  # [N]

        valid_pairs = mask_positive.sum(dim=1)
        valid_mask = valid_pairs > 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        # 5) 防止 0 / 0, inf / inf 之类
        numerator = torch.clamp(numerator, min=1e-8)
        denominator = torch.clamp(denominator, min=1e-8)

        ratio = numerator[valid_mask] / denominator[valid_mask]
        # 理论上 ratio ∈ (0,1]，这里再防一手数值抖动
        ratio = torch.clamp(ratio, min=1e-8, max=1.0)

        if self.reduction == 'mean':
            loss = -torch.log(ratio).mean()
        else:
            loss = -torch.log(ratio)

        return loss
