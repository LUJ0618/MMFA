import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from baseline.WavLM import *

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

class ASPM(nn.Module):
    def __init__(self, input_dim):
        super(ASPM, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)  # U^T
        self.attention_vector = nn.Linear(input_dim, 1, bias=True)  # v^T

    def forward(self, x, mask_ratio=0.5):
        """
        Args:
            x: Tensor of shape (batch_size, time_steps, feature_dim)
            mask_ratio: Float, proportion of frames to mask (e.g., 0.3 for 30%)
        Returns:
            masked_output: Tensor of shape (batch_size, time_steps, feature_dim)
        """
        # Linear transformation U^T * h_t + p
        scores = self.linear(x)  # Shape: (batch_size, time_steps, feature_dim)
        scores = torch.tanh(scores)  # Apply tanh activation

        # Attention score s_t = v^T * scores + q
        scores = self.attention_vector(scores).squeeze(-1)  # Shape: (batch_size, time_steps)

        # Sort scores to get indices of the lowest scores
        batch_size, time_steps = scores.shape
        num_mask_frames = int(time_steps * mask_ratio)

        sorted_scores, sorted_indices = torch.sort(scores, dim=1)
        mask_indices = sorted_indices[:, :num_mask_frames]  # Indices of the lowest scores

        # Create a mask tensor
        mask = torch.ones_like(scores, dtype=torch.float32)  # Start with all ones
        for i in range(batch_size):
            mask[i, mask_indices[i]] = 0  # Set the lowest-scoring frames to zero

        # Apply the mask: Set masked scores to a very small value (-inf)
        scores = torch.where(mask.bool(), scores, torch.full_like(scores, float('-inf')))

        # Softmax to compute attention weights α_t
        masked_attention_weights = F.softmax(scores, dim=1)  # Shape: (batch_size, time_steps)

        # 🔹 0 값 개수 계산 및 출력
        # 🔹 행별(프레임별)로 0 개수를 계산하여 출력
        # zero_counts_per_row = (masked_attention_weights == 0).sum(dim=1)

        # for row_idx, zero_count in enumerate(zero_counts_per_row.tolist()):  # `.tolist()`로 변환하여 출력
        #     print(f"🔹 Speaker {row_idx}: {zero_count} zero values")
            
        # num_zeros = (masked_attention_weights == 0).sum().item()
        # print(f"🔹 Masked Attention Weights - Zero Count: {num_zeros}")

        # Multiply attention weights with the input embeddings
        masked_output = x * masked_attention_weights.unsqueeze(-1)  # Shape: (batch_size, time_steps, feature_dim)

        return masked_output, masked_attention_weights

class ASPMSoftmax(nn.Module):
    def __init__(self, input_dim):
        super(ASPMSoftmax, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)  # U^T
        self.attention_vector = nn.Linear(input_dim, 1, bias=True)  # v^T

    def forward(self, x, mask_ratio=0.7):
        """
        Args:
            x: Tensor of shape (batch_size, time_steps, feature_dim)
            mask_ratio: Float, proportion of frames to mask (e.g., 0.3 for 30%)
        Returns:
            masked_output: Tensor of shape (batch_size, time_steps, feature_dim)
        """
        # Linear transformation U^T * h_t + p
        scores = self.linear(x)  # Shape: (batch_size, time_steps, feature_dim)
        scores = torch.tanh(scores)  # Apply tanh activation

        # Attention score s_t = v^T * scores + q
        scores = self.attention_vector(scores).squeeze(-1)  # Shape: (batch_size, time_steps)

        # Softmax to compute attention weights α_t
        attention_weights = F.softmax(scores, dim=1)  # Shape: (batch_size, time_steps)

        # Sort scores to get indices of the lowest scores (for masking)
        batch_size, time_steps = scores.shape
        num_mask_frames = int(time_steps * mask_ratio)

        sorted_scores, sorted_indices = torch.sort(scores, dim=1)
        mask_indices = sorted_indices[:, :num_mask_frames]  # Indices of the lowest scores

        # Create a mask tensor (default: all ones)
        mask = torch.ones_like(attention_weights, dtype=torch.float32)  # Shape: (batch_size, time_steps)
        for i in range(batch_size):
            mask[i, mask_indices[i]] = 0  # Set the lowest-scoring frames to zero

        # Apply the mask: Set masked attention weights to zero
        masked_attention_weights = attention_weights * mask  # Shape: (batch_size, time_steps)

        # Normalize again to ensure sum of attention weights is valid (prevent division by zero)
        # masked_attention_weights = masked_attention_weights / masked_attention_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        # Multiply attention weights with the input embeddings
        masked_output = x * masked_attention_weights.unsqueeze(-1)  # Shape: (batch_size, time_steps, feature_dim)

        return masked_output, masked_attention_weights

class ASPMSoftMasking(nn.Module):
    def __init__(self, input_dim):
        super(ASPMSoftMasking, self).__init__()
        self.input_dim = input_dim

        # Linear transformation layers for attention score calculation
        self.linear = nn.Linear(input_dim, input_dim, bias=True)  # U^T
        self.attention_vector = nn.Linear(input_dim, 1, bias=True)  # v^T

        # Learnable vector H0 for soft masking
        self.H0 = nn.Parameter(torch.randn(input_dim))  # Learnable parameter for H0

    def forward(self, x, mask_ratio=0.7):
        """
        Args:
            x: Tensor of shape (batch_size, time_steps, feature_dim)
        Returns:
            masked_output: Tensor of shape (batch_size, time_steps, feature_dim)
            masked_attention_weights: Tensor of shape (batch_size, time_steps)
        """
        batch_size, time_steps, feature_dim = x.shape

        # Linear transformation U^T * h_t + p
        scores = self.linear(x)  # Shape: (batch_size, time_steps, feature_dim)
        scores = torch.tanh(scores)  # Apply tanh activation

        # Attention score s_t = v^T * scores + q
        scores = self.attention_vector(scores).squeeze(-1)  # Shape: (batch_size, time_steps)

        # Softmax to compute attention weights α_t
        attention_weights = F.softmax(scores, dim=1)  # Shape: (batch_size, time_steps)

        # Sort scores to get indices of the lowest scores (for masking)
        num_mask_frames = int(time_steps * mask_ratio)
        sorted_scores, sorted_indices = torch.sort(scores, dim=1)
        mask_indices = sorted_indices[:, :num_mask_frames]  # Indices of the lowest scores

        # Create a mask tensor (default: all ones)
        mask = torch.ones_like(attention_weights, dtype=torch.float32)  # Shape: (batch_size, time_steps)
        for i in range(batch_size):
            mask[i, mask_indices[i]] = 0  # Set the lowest-scoring frames to zero

        # Apply the mask to attention weights (Soft Masking)
        masked_attention_weights = attention_weights * mask  # Shape: (batch_size, time_steps)

        # Soft Masking: Apply learnable vector H0 when mask is 0
        H0_expanded = self.H0.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, feature_dim)
        masked_output = x * masked_attention_weights.unsqueeze(-1) + (1 - masked_attention_weights.unsqueeze(-1)) * H0_expanded

        return masked_output, masked_attention_weights

class ASPMResidual(nn.Module):
    def __init__(self, input_dim):
        super(ASPMResidual, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)  # U^T
        self.attention_vector = nn.Linear(input_dim, 1, bias=True)  # v^T

    def forward(self, x, mask_ratio=0.5):
        """
        Args:
            x: Tensor of shape (batch_size, time_steps, feature_dim)
            mask_ratio: Float, proportion of frames to mask (e.g., 0.3 for 30%)
        Returns:
            residual_output: Tensor of shape (batch_size, time_steps, feature_dim)
        """
        # Linear transformation U^T * h_t + p
        scores = self.linear(x)  # Shape: (batch_size, time_steps, feature_dim)
        scores = torch.tanh(scores)  # Apply tanh activation

        # Attention score s_t = v^T * scores + q
        scores = self.attention_vector(scores).squeeze(-1)  # Shape: (batch_size, time_steps)

        # Softmax to compute attention weights α_t
        attention_weights = F.softmax(scores, dim=1)  # Shape: (batch_size, time_steps)

        # Sort scores to get indices of the lowest scores (for masking)
        batch_size, time_steps = scores.shape
        num_mask_frames = int(time_steps * mask_ratio)

        sorted_scores, sorted_indices = torch.sort(scores, dim=1)
        mask_indices = sorted_indices[:, :num_mask_frames]  # Indices of the lowest scores

        # Create a mask tensor (default: all ones)
        mask = torch.ones_like(attention_weights, dtype=torch.float32)  # Shape: (batch_size, time_steps)
        for i in range(batch_size):
            mask[i, mask_indices[i]] = 0  # Set the lowest-scoring frames to zero

        # Apply the mask: Set masked attention weights to zero
        masked_attention_weights = attention_weights * mask  # Shape: (batch_size, time_steps)

        # Residual 방식 적용: 기본 정보 유지하면서 가중치 반영
        residual_output = x + masked_attention_weights.unsqueeze(-1) * x  # Shape: (batch_size, time_steps, feature_dim)

        return residual_output, masked_attention_weights

class ASP(nn.Module):
    def __init__(self, input_dim):
        super(ASP, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)  # U^T
        self.attention_vector = nn.Linear(input_dim, 1, bias=True)  # v^T

    def forward(self, x):
        """
        Args:
            x: (batch_size, time_steps, feature_dim)
        Returns:
            pooled_output: (batch_size, feature_dim)
            attention_weights: (batch_size, time_steps)
        """
        # 1) 선형 변환 및 tanh
        scores = self.linear(x)               # (B, T, D)
        scores = torch.tanh(scores)           # (B, T, D)

        # 2) 스칼라 어텐션 스코어 계산 (v^T)
        scores = self.attention_vector(scores).squeeze(-1)  # (B, T)

        # 3) Softmax 로 프레임별 가중치 산출
        attention_weights = F.softmax(scores, dim=1)        # (B, T)

        # 4) 가중합 (Weighted Sum) → (B, D)
        pooled_output = torch.sum(
            x * attention_weights.unsqueeze(-1), 
            dim=1
        )

        return pooled_output, attention_weights

class ASPMRandom(nn.Module):
    def __init__(self, input_dim):
        super(ASPMRandom, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)  # U^T
        self.attention_vector = nn.Linear(input_dim, 1, bias=True)  # v^T

    def forward(self, x, mask_ratio=0.5):
        """
        Args:
            x: Tensor of shape (batch_size, time_steps, feature_dim)
            mask_ratio: Float, 전체 프레임 중 마스킹할 비율 (예: 0.3이면 30%를 무작위로 마스킹)
        Returns:
            masked_output: Tensor of shape (batch_size, time_steps, feature_dim)
            masked_attention_weights: Tensor of shape (batch_size, time_steps)
        """
        # 1) 선형 변환과 tanh 적용
        scores = self.linear(x)                    # (batch_size, time_steps, feature_dim)
        scores = torch.tanh(scores)                # tanh 활성화
        scores = self.attention_vector(scores).squeeze(-1)  # (batch_size, time_steps)

        # 2) 무작위 마스킹할 인덱스 선택
        batch_size, time_steps = scores.shape
        num_mask_frames = int(time_steps * mask_ratio)  # 마스킹할 프레임 수

        # 모든 위치를 1로 초기화한 mask 텐서
        mask = torch.ones_like(scores, dtype=torch.float32)  # (batch_size, time_steps)

        for i in range(batch_size):
            # randperm을 이용하여 무작위 프레임 인덱스 추출
            rand_indices = torch.randperm(time_steps)[:num_mask_frames]
            mask[i, rand_indices] = 0  # 마스킹할 위치에 0을 설정

        # 3) 소프트맥스 → 어텐션 가중치 계산
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, time_steps)

        # 4) 마스킹이 0인 곳에 어텐션 가중치 0으로 덮어씌우기
        masked_attention_weights = attention_weights * mask  # (batch_size, time_steps)

        # 5) 입력 x에 마스킹된 어텐션 가중치 적용
        masked_output = x * masked_attention_weights.unsqueeze(-1)  # (batch_size, time_steps, feature_dim)

        return masked_output, masked_attention_weights

class SlidingWindowAP(nn.Module):
    def __init__(self, window_size=5):
        super(SlidingWindowAP, self).__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=window_size, stride=1, padding=self.padding)

    def forward(self, x):
        # input x : batch, timestep, feature dim
        x = x.transpose(1,2)
        x = self.avg_pool(x)
        x = x.transpose(1, 2)

        return x

class LayerWiseBaseSlidingPool(nn.Module):
    def __init__(self, input_dim, num_layers, mask_ratio=0.0):
        super(LayerWiseBaseSlidingPool, self).__init__()
        self.mask_ratio = mask_ratio  # Proportion of frames to mask
        self.num_layers = num_layers  # Number of layers to handle
        self.layer_weights = nn.Parameter(torch.ones(num_layers))  # Learnable weights for each layer
        self.aspm_modules = nn.ModuleList([ASPMSoftmax(input_dim=input_dim) for _ in range(num_layers)])
        self.sliding_pool = SlidingWindowAP(window_size=5)

    def forward(self, layer_reps):
        """
        Args:
            layer_reps (torch.Tensor): Tensor of shape [Batch, Layer, Frame, Dim],
                                       containing the embeddings for each layer.
        Returns:
            torch.Tensor: Utterance-level embedding of shape [Batch, Dim].
        """
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)

        batch_size, num_layers, frame_size, dim = layer_reps.shape
        masked_outputs = []
        attn_weights = []

        for layer_idx in range(num_layers):
            # Get the embeddings for the current layer
            layer_output = layer_reps[:, layer_idx, :, :]  # Shape: [Batch, Frame, Dim]

            # Generate masked output for the current layer using its own ASPM module
            masked_output, attention_map = self.aspm_modules[layer_idx](layer_output, mask_ratio=self.mask_ratio)  # Shape: [Batch, Frame, Dim]
            sliding_pooled_output = self.sliding_pool(masked_output)

            # Append the masked output
            masked_outputs.append(sliding_pooled_output)
            attn_weights.append(attention_map)

        # self.get_attention_map(attn_weights)
        # Stack masked outputs and apply learnable weights
        masked_outputs = torch.stack(masked_outputs, dim=1)  # Shape: [Batch, Layer, Frame, Dim]
        weighted_outputs = (masked_outputs * self.layer_weights.view(1, -1, 1, 1)).sum(dim=1)  # Shape: [Batch, Frame, Dim]

        # Utterance-level average pooling
        utterance_level_embedding = torch.mean(weighted_outputs, dim=1)  # Shape: [Batch, Dim]

        return utterance_level_embedding

class LayerWiseASP(nn.Module):
    def __init__(self, input_dim, num_layers, mask_ratio=0.5):
        super(LayerWiseASP, self).__init__()
        self.mask_ratio = mask_ratio  # Proportion of frames to mask
        self.num_layers = num_layers  # Number of layers to handle
        self.layer_weights = nn.Parameter(torch.ones(num_layers))  # Learnable weights for each layer
        self.aspm_modules = nn.ModuleList([ASPM(input_dim=input_dim) for _ in range(num_layers)])
        self.final_attention_vector = nn.Linear(input_dim, 1, bias=True)  # v_t for utterance-level pooling
        self.final_linear = nn.Linear(input_dim, input_dim, bias=True)  # U_t for utterance-level pooling

    def forward(self, layer_reps):
        """
        Args:
            layer_reps (torch.Tensor): Tensor of shape [Batch, Layer, Frame, Dim],
                                       containing the embeddings for each layer.
        Returns:
            torch.Tensor: Utterance-level embedding of shape [Batch, Dim].
        """
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)

        batch_size, num_layers, frame_size, dim = layer_reps.shape
        masked_outputs = []

        for layer_idx in range(num_layers):
            # Get the embeddings for the current layer
            layer_output = layer_reps[:, layer_idx, :, :]  # Shape: [Batch, Frame, Dim]

            # Generate masked output for the current layer using its own ASPM module
            masked_output = self.aspm_modules[layer_idx](layer_output, mask_ratio=self.mask_ratio)  # Shape: [Batch, Frame, Dim]

            # Append the masked output
            masked_outputs.append(masked_output)

        # Stack masked outputs and apply learnable weights
        masked_outputs = torch.stack(masked_outputs, dim=1)  # Shape: [Batch, Layer, Frame, Dim]
        weighted_outputs = (masked_outputs * self.layer_weights.view(1, -1, 1, 1)).sum(dim=1)  # Shape: [Batch, Frame, Dim]

        # Utterance-level attention pooling
        scores = self.final_linear(weighted_outputs)  # Shape: [Batch, Frame, Dim]
        scores = torch.tanh(scores)  # Apply tanh activation
        scores = self.final_attention_vector(scores).squeeze(-1)  # Shape: [Batch, Frame]

        attention_weights = F.softmax(scores, dim=1)  # Shape: [Batch, Frame]
        utterance_level_embedding = torch.sum(weighted_outputs * attention_weights.unsqueeze(-1), dim=1)  # Shape: [Batch, Dim]

        return utterance_level_embedding

class LayerWiseBasePooling(nn.Module):
    """
    - 레이어별로 어텐티브 풀링 (마스킹 없음) 수행
    - 레이어별 학습 가중치 layer_weights로 합산
    - 최종 (B, D) 임베딩 출력
    """
    def __init__(self, input_dim, num_layers, mask_ratio=0.0):
        super(LayerWiseBasePooling, self).__init__()
        self.num_layers = num_layers
        self.mask_ratio = mask_ratio

        # 레이어별 학습 가중치
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # 각 레이어에 대응하는 AttentivePooling 모듈
        self.attentive_poolers = nn.ModuleList(
            [ASP(input_dim) for _ in range(num_layers)]
        )

    def forward(self, layer_reps):
        """
        Args:
            layer_reps: (Batch, Layer, Frame, Dim)
        Returns:
            utterance_level_embedding: (Batch, Dim)
        """
        # 만약 list 형태라면 tensor로 변환
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)
        # shape: (B, num_layers, F, D)

        batch_size, num_layers, frame_size, dim = layer_reps.shape

        pooled_outputs = []     # 레이어별 (B, D)
        attention_maps = []     # 레이어별 (B, F)

        # 1) 각 레이어별 어텐티브 풀링
        for layer_idx in range(num_layers):
            x_layer = layer_reps[:, layer_idx, :, :]  # (B, F, D)
            pooled, attn_map = self.attentive_poolers[layer_idx](x_layer)
            # pooled: (B, D)
            # attn_map: (B, F)

            pooled_outputs.append(pooled)
            attention_maps.append(attn_map)

        # 2) 레이어 풀링 결과 스택: (B, num_layers, D)
        pooled_outputs = torch.stack(pooled_outputs, dim=1)

        # 3) 레이어 가중치 곱 → 합산
        #    layer_weights: (num_layers,) → (1, num_layers, 1)
        #    => 브로드캐스팅 곱 후 sum(dim=1) => (B, D)
        weighted_sum = (
            pooled_outputs
            * self.layer_weights.view(1, -1, 1)
        ).sum(dim=1)

        # (B, D)
        utterance_level_embedding = weighted_sum

        return utterance_level_embedding

class LayerWiseWeightScale(nn.Module):
    def __init__(self, input_dim, num_layers, mask_ratio=0.0, scale_factor_init=1.0, scale_factor_min=0.1, scale_factor_max=10.0):
        super(LayerWiseWeightScale, self).__init__()
        self.mask_ratio = mask_ratio  # Proportion of frames to mask
        self.num_layers = num_layers  # Number of layers to handle

        # (1) 학습 가능한 레이어 가중치 (Softmax 정규화 대상)
        self.layer_weights = nn.Parameter(torch.ones(num_layers))  # 초기값 1

        # (2) 학습 가능한 Scale Factor (초기값 설정 및 클리핑 범위 지정)
        self.scale_factor = nn.Parameter(torch.tensor(scale_factor_init))  # 학습 가능한 Scale Factor 초기값
        self.scale_factor_min = scale_factor_min
        self.scale_factor_max = scale_factor_max

        # (3) 레이어별 ASPM 모듈 (랜덤 마스킹 사용 예시)
        self.aspm_modules = nn.ModuleList([ASPMSoftmax(input_dim=input_dim) for _ in range(num_layers)])

    def forward(self, layer_reps):
        """
        Args:
            layer_reps (torch.Tensor): Tensor of shape [Batch, Layer, Frame, Dim],
                                       containing the embeddings for each layer.
        Returns:
            torch.Tensor: Utterance-level embedding of shape [Batch, Dim].
        """
        # 0) 입력 형태가 list라면 [Batch, Layer, Frame, Dim]으로 변환
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)

        batch_size, num_layers, frame_size, dim = layer_reps.shape
        masked_outputs = []
        attn_weights = []

        # 1) 레이어별 ASPM 모듈을 통해 마스킹된 출력과 어텐션 맵 계산
        for layer_idx in range(num_layers):
            layer_output = layer_reps[:, layer_idx, :, :]  # [Batch, Frame, Dim]
            masked_output, attention_map = self.aspm_modules[layer_idx](layer_output, mask_ratio=self.mask_ratio)
            masked_outputs.append(masked_output)
            attn_weights.append(attention_map)

        # 2) 마스킹된 출력들을 [Batch, Layer, Frame, Dim] 형태로 스택
        masked_outputs = torch.stack(masked_outputs, dim=1)

        # 3) 학습 가능한 Scale Factor를 곱한 Softmax 정규화된 레이어 가중치 계산
        normalized_layer_weights = F.softmax(self.layer_weights, dim=0) * self.scale_factor

        # 4) 레이어별 가중합을 계산 (Weighted Sum)
        weighted_outputs = (masked_outputs * normalized_layer_weights.view(1, -1, 1, 1)).sum(dim=1)  # [Batch, Frame, Dim]

        # 5) Utterance-level 평균 풀링 (Average Pooling)
        utterance_level_embedding = torch.mean(weighted_outputs, dim=1)  # [Batch, Dim]

        # 6) Gradient Clipping을 통해 학습 가능한 Scale Factor의 값 제한
        with torch.no_grad():
            self.scale_factor.data = torch.clamp(self.scale_factor, min=self.scale_factor_min, max=self.scale_factor_max)

        return utterance_level_embedding

class LayerWiseBase(nn.Module):
    def __init__(self, input_dim, num_layers, mask_ratio=0.0):
        super(LayerWiseBase, self).__init__()
        self.mask_ratio = mask_ratio  # Proportion of frames to mask
        self.num_layers = num_layers  # Number of layers to handle
        self.layer_weights = nn.Parameter(torch.ones(num_layers))  # Learnable weights for each layer
        self.aspm_modules = nn.ModuleList([ASPMSoftmax(input_dim=input_dim) for _ in range(num_layers)])

    def forward(self, layer_reps):
        """s
        Args:
            layer_reps (torch.Tensor): Tensor of shape [Batch, Layer, Frame, Dim],
                                       containing the embeddings for each layer.
        Returns:
            torch.Tensor: Utterance-level embedding of shape [Batch, Dim].
        """
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)

        batch_size, num_layers, frame_size, dim = layer_reps.shape
        masked_outputs = []
        attn_weights = []

        for layer_idx in range(num_layers):
            # Get the embeddings for the current layer
            layer_output = layer_reps[:, layer_idx, :, :]  # Shape: [Batch, Frame, Dim]

            # Generate masked output for the current layer using its own ASPM module
            masked_output, attention_map = self.aspm_modules[layer_idx](layer_output, mask_ratio=self.mask_ratio)  # Shape: [Batch, Frame, Dim]

            # Append the masked output
            masked_outputs.append(masked_output)
            attn_weights.append(attention_map)

        # self.get_attention_map(attn_weights)
        # Stack masked outputs and apply learnable weights
        masked_outputs = torch.stack(masked_outputs, dim=1)  # Shape: [Batch, Layer, Frame, Dim]
        weighted_outputs = (masked_outputs * self.layer_weights.view(1, -1, 1, 1)).sum(dim=1)  # Shape: [Batch, Frame, Dim]

        # Utterance-level average pooling
        utterance_level_embedding = torch.mean(weighted_outputs, dim=1)  # Shape: [Batch, Dim]

        return utterance_level_embedding

    def get_attention_map(self, attention_maps, save_dir="./attention_maps_softmax"):
        """ 
        🔹 어텐션 맵 (Heatmap) + 레이어별 1D 그래프를 하나의 피규어로 저장하는 함수
        Args:
            attention_maps: (Batch, Layers, Time Steps)
            save_dir: 저장할 디렉토리
        """
        os.makedirs(save_dir, exist_ok=True)  # 🔥 저장할 폴더 생성

        attention_maps = torch.stack(attention_maps, dim=1).detach()  # (Batch, Layers, Time Steps, 1)
        batch_size, num_layers, time_steps = attention_maps.shape

        for batch_idx in range(batch_size):  # 🔥 배치별로 저장
            avg_attention = attention_maps[batch_idx].squeeze(-1)  # (Layers, Time Steps)

            fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 2]})  # 🔥 히트맵: 선 그래프 = 3:2 비율

            ## 🔹 (1) 어텐션 맵 히트맵 (위)
            sns.heatmap(avg_attention.cpu().numpy(), cmap="Reds", ax=axes[0], square="auto")
            axes[0].set_xlabel("Time Steps", fontsize=10)
            axes[0].set_ylabel("Layers", fontsize=10)
            axes[0].set_title(f"Attention Map (Sample {batch_idx})", fontsize=12)

            ## 🔹 (2) 어텐션 값 1D 선 그래프 (아래)
            for layer_idx in range(num_layers):
                axes[1].plot(
                    range(time_steps), avg_attention[layer_idx].cpu().numpy(), label=f"Layer {layer_idx}"
                )

            axes[1].set_xlabel("Time Steps", fontsize=10)
            axes[1].set_ylabel("Attention Weight", fontsize=10)
            axes[1].set_title("Layer-wise Attention Distribution over Time", fontsize=12)
            axes[1].legend(loc="upper right", fontsize=8)
            axes[1].grid(True)

            ## 🔹 저장
            save_path = os.path.join(save_dir, f"attn_map_sample_{batch_idx}.png")
            plt.tight_layout()  # 레이아웃 조정
            plt.savefig(save_path, dpi=300)  # 🔥 저장
            plt.close()

            print(f"✅ 어텐션 맵 + 1D 그래프 저장 완료: {save_path}")

class LayerWiseNone(nn.Module):
    def __init__(self, num_layers, mask_ratio=0.0):
        super(LayerWiseNone, self).__init__()
        self.mask_ratio = mask_ratio
        self.num_layers = num_layers
        # 레이어별 학습 가중치 (초기값 1)
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, layer_reps):
        """
        Args:
            layer_reps (torch.Tensor or list):
                - [Batch, Layer, Frame, Dim] 형태의 텐서
                  (레이어 수 = self.num_layers)
                - 만약 list 형태라면, [Batch, Layer, Frame, Dim]으로 stack 처리
        Returns:
            torch.Tensor: [Batch, Dim] 형태의 최종 임베딩
        """
        # 만약 layer_reps가 list로 들어온다면 tensor로 변환
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)
        
        # layer_reps.shape: (Batch, num_layers, Frame, Dim)

        # 1) 레이어별 가중치 곱 + 합산
        #    layer_weights (num_layers,) → (1, num_layers, 1, 1)
        #    => 레이어 차원(num_layers)에 대해 브로드캐스팅 곱
        weighted_sum = (
            layer_reps * self.layer_weights.view(1, -1, 1, 1)
        ).sum(dim=1)  # 결과 shape: [Batch, Frame, Dim]

        # 2) 프레임 차원에 대해 평균 풀링 -> [Batch, Dim]
        utterance_level_embedding = weighted_sum.mean(dim=1)

        return utterance_level_embedding

class LayerWiseNoneSelect(nn.Module):
    def __init__(self, num_layers, mask_ratio=0.0):
        super(LayerWiseNoneSelect, self).__init__()
        self.mask_ratio = mask_ratio
        self.num_layers = num_layers
        
        # 레이어별 학습 가중치 (초기값 1)
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # 내부적으로 여러 레이어 인덱스 지정 (0-based)
        # 예: [5, 6, 7] → 사람 기준(1-based)으론 6,7,8번째 레이어
        self.selected_layers = [5, 6, 7, 10, 12]

    def forward(self, layer_reps):
        """
        Args:
            layer_reps (torch.Tensor or list):
                - [Batch, Layer, Frame, Dim] 형태 (num_layers = self.num_layers)
                - list라면 stack하여 모양 통일
        Returns:
            torch.Tensor: [Batch, Dim] 형태 최종 임베딩
        """
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)
        # => (B, num_layers, F, D)

        # (1) 여러 레이어를 선택해서 추출
        #     (B, selected_layer_count, F, D) 형태
        selected_out = layer_reps[:, self.selected_layers, :, :]
        
        # (2) 선택된 레이어의 가중치만 가져옴
        #     (num_layers,) → (selected_layer_count,)
        selected_weights = self.layer_weights[self.selected_layers]

        # (3) 레이어별 가중치 곱
        #     selected_out.shape = (B, selected_layer_count, F, D)
        #     selected_weights.shape = (selected_layer_count,)
        #     브로드캐스팅 위해 (1, selected_layer_count, 1, 1)
        weighted_sum = (
            selected_out * selected_weights.view(1, -1, 1, 1)
        ).sum(dim=1)  # (B, F, D)

        # (4) 프레임 차원 평균 풀링 → (B, D)
        utterance_level_embedding = weighted_sum.mean(dim=1)

        return utterance_level_embedding

class LayerWiseNoneAttn(nn.Module):
    def __init__(self, input_dim, num_layers, mask_ratio=0.0):
        super(LayerWiseNoneAttn, self).__init__()
        self.mask_ratio = mask_ratio  # 여기서는 사용 안 함
        self.num_layers = num_layers

        # (1) 레이어별 학습 가중치 (초기값 1)
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # (2) 최종 프레임 풀링을 위한 어텐션 파트(Attentive Pooling)
        self.att_linear = nn.Linear(input_dim, input_dim, bias=True)    # U^T
        self.att_vector = nn.Linear(input_dim, 1, bias=True)           # v^T

    def forward(self, layer_reps):
        """
        Args:
            layer_reps (torch.Tensor or list):
                - [Batch, num_layers, Frame, Dim] 형태 텐서
                - list로 들어오면 stack하여 모양 통일
        Returns:
            utterance_level_embedding: [Batch, Dim] 최종 임베딩
        """
        # 0) 만약 list라면 [Batch, Layer, Frame, Dim]으로 변환
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)
        # => shape: (B, num_layers, Frame, Dim)

        # 1) 레이어별 가중치 곱 + 합산
        #    layer_weights: (num_layers,) → (1, num_layers, 1, 1)로 reshape
        #    레이어 차원(num_layers) 기준으로 합산하면 -> (B, Frame, Dim)
        weighted_sum = (
            layer_reps * self.layer_weights.view(1, -1, 1, 1)
        ).sum(dim=1)  # shape: [Batch, Frame, Dim]

        # 2) 마지막 프레임 풀링을 어텐티브 풀링으로 수행
        #    (B, Frame, Dim) → (B, Dim)
        # 2-1) 점수 계산: Linear → tanh → Linear
        scores = self.att_linear(weighted_sum)         # (B, F, D)
        scores = torch.tanh(scores)
        scores = self.att_vector(scores).squeeze(-1)   # (B, F)

        # 2-2) Softmax로 프레임별 가중치 계산
        attention_weights = F.softmax(scores, dim=1)    # (B, F)

        # 2-3) 가중합(Weighted Sum)
        #      (B, F, D) * (B, F) -> (B, D)
        utterance_level_embedding = torch.sum(
            weighted_sum * attention_weights.unsqueeze(-1), 
            dim=1
        )
        # self.save_attention_map(attention_weights)

        return utterance_level_embedding

    def save_attention_map(self, attention_weights):
        """
        🔹 단일 어텐션 맵을 시각화하고 저장하는 함수
        Args:
            attention_weights: (Batch, Time Steps)
        """
        save_dir = "after_attn_map"
        os.makedirs(save_dir, exist_ok=True)  # 🔥 저장할 폴더 생성

        batch_size, time_steps = attention_weights.shape

        for batch_idx in range(batch_size):
            attn_map = attention_weights[batch_idx].cpu().detach().numpy()  # (Time Steps,)

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 2]})

            ## 🔹 (1) 어텐션 맵 히트맵 (위)
            sns.heatmap(attn_map[None, :], cmap="Reds", ax=axes[0], cbar=True, xticklabels=50, yticklabels=[])
            axes[0].set_xlabel("Time Steps", fontsize=10)
            axes[0].set_title(f"Attention Map (Sample {batch_idx})", fontsize=12)

            ## 🔹 (2) 어텐션 값 1D 선 그래프 (아래)
            axes[1].plot(range(time_steps), attn_map, label=f"Attention Weight", color='red')
            axes[1].set_xlabel("Time Steps", fontsize=10)
            axes[1].set_ylabel("Attention Weight", fontsize=10)
            axes[1].set_title("Frame-wise Attention Distribution", fontsize=12)
            axes[1].grid(True)

            ## 🔹 저장
            save_path = os.path.join(save_dir, f"attn_map_sample_{batch_idx}.png")
            plt.tight_layout()  # 레이아웃 조정
            plt.savefig(save_path, dpi=300)  # 🔥 저장
            plt.close()

            print(f"✅ 어텐션 맵 저장 완료: {save_path}")

class LayerWiseSelect(nn.Module):
    def __init__(self, input_dim, num_layers, mask_ratio=0.0):
        super(LayerWiseSelect, self).__init__()
        self.mask_ratio = mask_ratio
        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.ones(num_layers))  
        self.aspm_modules = nn.ModuleList(
            [ASPM(input_dim=input_dim) for _ in range(num_layers)]
        )

    def forward(self, layer_reps):
        """
        Args:
            layer_reps (torch.Tensor): [Batch, Layer, Frame, Dim]
        Returns:
            torch.Tensor: [Batch, Dim] 발화(utterance) 임베딩
        """
        # layer_reps가 list라면, [Batch, Layer, Frame, Dim] 형태로 스택
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)

        # 예: 특정 레이어만 선택해서 사용
        # 5 6 7 10 12
        selected_layers = [5, 6, 7]
        # ※ 실제로는 6, 7, 8 레이어가 존재하는지(num_layers > 8) 체크가 필요할 수 있음.

        # 선택된 레이어의 masked_output/attention_map 저장
        selected_outputs = []
        selected_attn_weights = []

        for layer_idx in selected_layers:
            # 해당 레이어 임베딩
            layer_output = layer_reps[:, layer_idx, :, :]  # [Batch, Frame, Dim]

            # 무작위 마스킹 ASPM 모듈 적용
            masked_output, attention_map = self.aspm_modules[layer_idx](
                layer_output, mask_ratio=self.mask_ratio
            )
            selected_outputs.append(masked_output)
            selected_attn_weights.append(attention_map)

        # [Batch, #selected_layers, Frame, Dim] 형태로 합치기
        selected_outputs = torch.stack(selected_outputs, dim=1)

        # 선택된 레이어의 가중치만 골라옴
        selected_layer_weights = self.layer_weights[selected_layers]  
        # [num_layers] → [#selected_layers]

        # 레이어 가중치 곱해준 뒤 sum(레이어 차원)
        weighted_outputs = (selected_outputs *
                            selected_layer_weights.view(1, -1, 1, 1)).sum(dim=1)
        # 결과: [Batch, Frame, Dim]
        # 최종 프레임 차원 평균 → [Batch, Dim]
        utterance_level_embedding = torch.mean(weighted_outputs, dim=1)

        return utterance_level_embedding

class LayerWiseBaseEpoch(nn.Module):
    def __init__(self, input_dim, num_layers, total_epochs=15, mask_ratio=0.0):
        super(LayerWiseBaseEpoch, self).__init__()
        self.num_layers = num_layers
        self.total_epochs = total_epochs
        self.current_epoch = 0  # 🔥 학습 시작 시 epoch을 0으로 초기화
        self.layer_weights = nn.Parameter(torch.ones(num_layers))  # 학습 가능한 레이어 가중치
        self.aspm_modules = nn.ModuleList([ASPM(input_dim=input_dim) for _ in range(num_layers)])

        # 초반에는 마스킹 없음 (0%) → 최종적으로 50%까지 증가
        self.initial_mask_ratio = 0.0  # 🔥 0% (초반 5 epoch 동안)
        self.increase_epochs = 10
        self.mask_ratio = mask_ratio      # 🔥 최종 50%

    def update_epoch(self):
        """ 🔥 외부에서 호출하여 현재 epoch 값을 업데이트하는 함수 """
        self.current_epoch = min(self.current_epoch + 1, self.total_epochs)

    def get_mask_ratios(self):
        """ 🔥 초반 10 epoch 동안 50%까지 증가, 이후 고정 """
        if self.current_epoch < self.increase_epochs:
            # 🔥 10 epoch 동안 선형적으로 증가
            progress = self.current_epoch / self.increase_epochs
            mask_ratio = self.initial_mask_ratio + (self.mask_ratio - self.initial_mask_ratio) * progress
        else:
            # 🔥 10 epoch 이후에는 50% 고정
            mask_ratio = self.mask_ratio
       
        return [mask_ratio] * self.num_layers  # 모든 레이어에 동일한 비율 적용


    def forward(self, layer_reps):
        """ 🔥 forward에서 epoch을 따로 입력하지 않아도 됨 """
        if isinstance(layer_reps, list):
            layer_reps = torch.stack(layer_reps, dim=1)

        batch_size, num_layers, frame_size, dim = layer_reps.shape
        masked_outputs = []

        mask_ratios = self.get_mask_ratios()  # 🔥 내부적으로 epoch을 참조하여 마스킹 비율 계산

        
        for layer_idx in range(num_layers):
            layer_output = layer_reps[:, layer_idx, :, :]
            mask_ratio = mask_ratios[layer_idx]

            masked_output = self.aspm_modules[layer_idx](layer_output, mask_ratio=self.mask_ratio)

            masked_outputs.append(masked_output)

        masked_outputs = torch.stack(masked_outputs, dim=1)

        weighted_sum = (masked_outputs * self.layer_weights.view(1, -1, 1, 1)).sum(dim=1)

        utterance_level_embedding = torch.mean(weighted_sum, dim=1)

        return utterance_level_embedding

class spk_extractor(nn.Module):
    def __init__(self, mask_ratio = 0.0, **kwargs):
        super(spk_extractor, self).__init__()

        print(f"🔹 [spk_extractor] Initialized with mask_ratio: {mask_ratio}")

        checkpoint = torch.load('baseline/pretrained/WavLM-Base+.pt')
        # print("Pre-trained Model: {}".format(kwargs['pretrained_model_path']))
        # checkpoint = torch.load(kwargs['pretrained_model_path'])
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])
        # Add LayerWiseASP module
        self.weighted_pooling = LayerWiseBase(input_dim=768, num_layers=13, mask_ratio=mask_ratio)
        self.final_linear = nn.Linear(768, 256, bias=True)  # Final projection layer


    def forward(self, wav_and_flag):
        
        x = wav_and_flag[0]

        cnn_outs, layer_results =  self.model.extract_features(x, output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        # x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)

        utterance_level_embedding = self.weighted_pooling(layer_reps)

        final_output = self.final_linear(utterance_level_embedding) 

        # out = self.backend(x)
        return final_output

    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            

            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);


def MainModel(**kwargs):
    mask_ratio = kwargs.get("mask_ratio", 0.0)
    print(f"🔹 [MainModel] Passing mask_ratio: {mask_ratio}")
    model = spk_extractor(**kwargs)
    return model

if __name__ == "__main__":
        
    from torchinfo import summary

    # Model initialization
    model = spk_extractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dummy input for testing, shape (1, 48240)
    dummy_input = torch.randn(1, 40, 48240).to(device)

    # Correct input format for the forward method
    summary(model, input_data=(dummy_input))
