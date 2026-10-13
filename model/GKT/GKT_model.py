"""GKT (Graph-based Knowledge Tracing) 模型实现

基于论文: "Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Networks"
参考实现: https://github.com/jhljx/GKT
"""

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MLP(nn.Module):
    """两层全连接 ReLU 网络带批归一化

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: Dropout 概率
        bias: 是否使用偏置
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs: torch.Tensor) -> torch.Tensor:
        """批归一化，处理 batch_size=1 或 3D 输入的情况"""
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 或 0 会导致 BatchNorm 错误，直接返回输入
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:
            return self.norm(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """擦除与添加门模块

    注意：这个擦除添加门与 DKVMN 中的略有不同。
    更多信息请参考论文 "Dynamic Key-Value Memory Networks for Knowledge Tracing"
    论文链接: https://arxiv.org/abs/1611.08108

    Args:
        feature_dim: 特征维度
        num_c: 概念数量
        bias: 是否使用偏置
    """

    def __init__(self, feature_dim: int, num_c: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(num_c))
        self.reset_parameters()
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征矩阵，形状为 [batch_size, num_c, feature_dim]

        Returns:
            处理后的特征矩阵，形状为 [batch_size, num_c, feature_dim]
        """
        erase_gate = torch.sigmoid(self.erase(x))  # [batch_size, num_c, feature_dim]
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))  # [batch_size, num_c, feature_dim]
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res


class GKT(nn.Module):
    """基于图的知识追踪模型

    使用图神经网络建模学生知识掌握程度

    Args:
        num_c: 概念（技能）数量
        hidden_dim: 隐藏层维度
        emb_size: 嵌入维度
        graph_type: 图类型，"dense" 或 "transition"
        graph: 预计算的图邻接矩阵，形状为 [num_c, num_c]
        dropout: Dropout 概率
        bias: 是否使用偏置
        use_gradient_checkpointing: 训练时是否启用逐时间步梯度检查点
    """

    def __init__(
        self,
        num_c: int,
        hidden_dim: int,
        emb_size: int,
        graph_type: str = "dense",
        graph: torch.Tensor = None,
        dropout: float = 0.5,
        bias: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_c = num_c
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.res_len = 2  # 响应类型数量 (0/1)
        self.graph_type = graph_type

        # 图是先验常量而非可学习参数，用 buffer 避免占用优化器槽位
        if graph is None:
            graph = self._build_default_graph(num_c, graph_type)
        self.register_buffer("graph", graph, persistent=True)

        # 概念和交互嵌入
        self.interaction_emb = nn.Embedding(self.res_len * num_c, emb_size)
        self.emb_c = nn.Embedding(num_c + 1, emb_size, padding_idx=-1)

        # 自身特征变换函数
        mlp_input_dim = hidden_dim + emb_size
        self.f_self = MLP(
            mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias
        )

        # 邻居特征变换函数列表
        self.f_neighbor_list = nn.ModuleList()
        # f_in: 入边特征变换
        self.f_neighbor_list.append(
            MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        )
        # f_out: 出边特征变换
        self.f_neighbor_list.append(
            MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        )

        # 擦除添加门
        self.erase_add_gate = EraseAddGate(hidden_dim, num_c)
        # GRU 单元
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # 预测层
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._bn_layers = [m for m in self.modules() if isinstance(m, nn.BatchNorm1d)]

    def _build_default_graph(self, num_c: int, graph_type: str) -> torch.Tensor:
        """构建默认图

        Args:
            num_c: 概念数量
            graph_type: 图类型

        Returns:
            图邻接矩阵
        """
        import numpy as np

        if graph_type == "dense":
            graph = 1.0 / (num_c - 1) * np.ones((num_c, num_c))
            np.fill_diagonal(graph, 0)
        else:
            # 默认使用单位矩阵作为后备
            graph = np.eye(num_c)
        return torch.from_numpy(graph).float()

    def _aggregate(
        self, xt: torch.Tensor, qt: torch.Tensor, ht: torch.Tensor
    ) -> torch.Tensor:
        """聚合步骤

        将当前交互嵌入写入对应概念行，再与隐状态拼接

        Args:
            xt: 当前时间步的交互特征索引，形状为 [batch_size]
            qt: 当前时间步的问题索引，形状为 [batch_size]
            ht: 当前时间步所有概念的隐藏表示，形状为 [batch_size, num_c, hidden_dim]

        Returns:
            聚合后的特征，形状为 [batch_size, num_c, hidden_dim + emb_size]
        """
        device = xt.device
        batch_size = ht.shape[0]

        # 概念嵌入展开到 batch；emb_c 第 num_c 行仅在无效交互时才会保留，
        # 当前数据下不会发生，故保持参考实现的 num_c+1 行不变。
        concept_embedding = self.emb_c(
            torch.arange(self.num_c, device=device).expand(batch_size, -1)
        )  # [batch_size, num_c, emb_size]

        # 将当前交互的嵌入放入 qt 指定的那一行
        concept_embedding = concept_embedding.index_put(
            (torch.arange(batch_size, device=device), qt.long()),
            self.interaction_emb(xt),
        )

        # 拼接隐藏状态和概念嵌入
        tmp_ht = torch.cat(
            (ht, concept_embedding), dim=-1
        )  # [batch_size, num_c, hidden_dim + emb_size]
        return tmp_ht

    def _agg_neighbors(self, tmp_ht: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        """邻居聚合步骤

        使用图神经网络聚合邻居信息

        Args:
            tmp_ht: 聚合后的特征，形状为 [batch_size, num_c, hidden_dim + emb_size]
            qt: 当前时间步的问题索引，形状为 [batch_size]

        Returns:
            聚合邻居后的特征，形状为 [batch_size, num_c, hidden_dim]
        """
        device = qt.device
        batch_size = tmp_ht.shape[0]

        # 获取自身特征
        self_index_tuple = (torch.arange(batch_size, device=device), qt.long())
        self_ht = tmp_ht[self_index_tuple]  # [batch_size, hidden_dim + emb_size]
        self_features = self.f_self(self_ht)  # [batch_size, hidden_dim]

        # 扩展自身特征用于邻居聚合
        expanded_self_ht = self_ht.unsqueeze(dim=1).expand(-1, self.num_c, -1)
        # [batch_size, num_c, hidden_dim + emb_size]
        neigh_ht = torch.cat((expanded_self_ht, tmp_ht), dim=-1)
        # [batch_size, num_c, 2 * (hidden_dim + emb_size)]

        # 获取邻接矩阵
        adj = self.graph[qt.long(), :].unsqueeze(dim=-1)  # [batch_size, num_c, 1]
        reverse_adj = self.graph[:, qt.long()].transpose(0, 1).unsqueeze(dim=-1)
        # [batch_size, num_c, 1]

        # 计算邻居特征
        m_next = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * (
            self.f_neighbor_list[1](neigh_ht)
        )  # [batch_size, num_c, hidden_dim]

        # 自身行只保留 self 分支，不叠加邻居信息
        return m_next.index_put(self_index_tuple, self_features)

    def _update(
        self, tmp_ht: torch.Tensor, ht: torch.Tensor, qt: torch.Tensor
    ) -> torch.Tensor:
        """更新步骤

        使用擦除添加门和 GRU 更新隐藏状态

        Args:
            tmp_ht: 聚合后的特征，形状为 [batch_size, num_c, hidden_dim + emb_size]
            ht: 当前隐藏状态，形状为 [batch_size, num_c, hidden_dim]
            qt: 当前时间步的问题索引，形状为 [batch_size]

        Returns:
            更新后的隐藏状态，形状为 [batch_size, num_c, hidden_dim]
        """
        # GNN 聚合
        m_next = self._agg_neighbors(tmp_ht, qt)  # [batch_size, num_c, hidden_dim]

        # 擦除添加门
        m_next = self.erase_add_gate(m_next)

        # GRU 更新
        res = self.gru(
            m_next.reshape(-1, self.hidden_dim), ht.reshape(-1, self.hidden_dim)
        )  # [batch_size * num_c, hidden_dim]
        return res.view(-1, self.num_c, self.hidden_dim)

    def _predict_step(self, h_next: torch.Tensor) -> torch.Tensor:
        """预测步骤

        预测所有概念的正确概率

        Args:
            h_next: 下一时间步的隐藏状态，形状为 [batch_size, num_c, hidden_dim]

        Returns:
            预测概率，形状为 [batch_size, num_c]
        """
        return torch.sigmoid(
            self.predict(h_next).squeeze(dim=-1)
        )  # [batch_size, num_c]

    def _get_next_pred(self, yt: torch.Tensor, q_next: torch.Tensor) -> torch.Tensor:
        """获取下一时间步的预测

        Args:
            yt: 所有概念的预测概率，形状为 [batch_size, num_c]
            q_next: 下一时间步的问题索引，形状为 [batch_size]

        Returns:
            下一问题的预测概率，形状为 [batch_size]
        """
        return yt.gather(1, q_next.clamp(max=self.num_c - 1).unsqueeze(1)).squeeze(1)

    def _step(
        self, ht: torch.Tensor, xt: torch.Tensor, qt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单个时间步：聚合 + 更新 + 预测

        Args:
            ht: 当前隐藏状态，形状为 [batch_size, num_c, hidden_dim]
            xt: 当前时间步的交互特征索引，形状为 [batch_size]
            qt: 当前时间步的问题索引，形状为 [batch_size]

        Returns:
            (h_next, yt)：新隐藏状态与全概念预测
        """
        tmp_ht = self._aggregate(xt, qt, ht)
        h_next = self._update(tmp_ht, ht, qt)
        yt = self._predict_step(h_next)
        return h_next, yt

    @contextlib.contextmanager
    def _frozen_bn_stats(self):
        # 检查点重算分支不得二次更新 BN running stats（原前向已更新）
        saved = [
            (
                bn,
                bn.momentum,
                bn.running_mean.clone(),
                bn.running_var.clone(),
                bn.num_batches_tracked.clone(),
            )
            for bn in self._bn_layers
        ]
        try:
            yield
        finally:
            for bn, momentum, running_mean, running_var, num_batches in saved:
                bn.momentum = momentum
                bn.running_mean.copy_(running_mean)
                bn.running_var.copy_(running_var)
                bn.num_batches_tracked.copy_(num_batches)

    def _checkpoint_contexts(self):
        return contextlib.nullcontext(), self._frozen_bn_stats()

    def forward(self, sequence: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """前向传播

        预测语义：y[:, t] 预测 response[:, t+1]
        即在时间步 t 处理完 (sequence[t], response[t]) 后，预测下一个位置 response[t+1]

        Args:
            sequence: 概念ID序列，形状为 [batch_size, sequence_length]
            response: 响应序列，形状为 [batch_size, sequence_length]

        Returns:
            预测结果，形状为 [batch_size, sequence_length - 1]
            y[:, t] 预测的是 response[:, t+1]
        """
        device = sequence.device
        batch_size, seq_len = sequence.shape

        # 初始化隐藏状态
        ht = torch.zeros((batch_size, self.num_c, self.hidden_dim), device=device)

        # 生成交互特征: q * 2 + r
        features = sequence * 2 + response
        questions = sequence

        pred_list = []
        for i in range(seq_len - 1):  # 只需要处理前 seq_len-1 个位置
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]

            if self.use_gradient_checkpointing and self.training:
                ht, yt = checkpoint(
                    self._step,
                    ht,
                    xt,
                    qt,
                    use_reentrant=False,
                    context_fn=self._checkpoint_contexts,
                )
            else:
                ht, yt = self._step(ht, xt, qt)

            # 获取下一时间步的预测（对 response[:, i+1] 的预测）
            pred = self._get_next_pred(yt, questions[:, i + 1])
            pred_list.append(pred)

        # 堆叠预测结果
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]

        return pred_res
