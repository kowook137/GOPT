
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, keys, values, pad_mask=None):
        # A.P.: Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #A.P.: Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # A.P.: (N, value_len, heads, head_dim)
        keys = self.keys(keys)        # A.P.: (N, key_len, heads, head_dim)
        queries = self.queries(query) # A.P.: (N, query_len, heads, heads_dim)

        # A.P.: Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # A.P.: queries shape: (N, query_len, heads, heads_dim),
        # A.P.: keys shape: (N, key_len, heads, heads_dim)
        # A.P.: energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).expand(N, query_len, key_len)
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            energy = energy.masked_fill(pad_mask==0, -1e18)
            # energy = energy.masked_fill(pad_mask==0, float("-inf"))

        # A.P.: Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # A.P.: attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # A.P.: attention shape: (N, heads, query_len, key_len)
        # A.P.: values shape: (N, value_len, heads, heads_dim)
        # A.P.: out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # A.P.: Linear layer doesn't modify the shape, final shape will be (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pad_mask=None):
        attention = self.attention(query, key, value, pad_mask)

        # A.P.: Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderBlock, self).__init__()
        
        self.item_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.ems_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.ems_on_item = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.item_on_ems = TransformerBlock(embed_size, heads, dropout, forward_expansion)

    def forward(self, item_feature, ems_feature, mask=None):
        # self-attention
        item_embedding = self.item_embedding(item_feature, item_feature, item_feature)
        ems_embedding = self.ems_embedding(ems_feature, ems_feature, ems_feature, mask)
        # cross-attention
        ems_on_item = self.ems_on_item(ems_embedding, item_embedding, item_embedding, mask) 
        item_on_ems = self.item_on_ems(item_embedding, ems_embedding, ems_embedding)

        return item_on_ems, ems_on_item
    

def _obs_get(obs: Any, key: str) -> Any:
    # Tianshou Batch와 dict 관측을 모두 같은 방식으로 읽기 위한 헬퍼다.
    if isinstance(obs, dict):
        return obs[key]
    if hasattr(obs, key):
        return getattr(obs, key)
    return obs[key]


def _to_tensor(
    value: Any,
    device: Union[str, int, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _ensure_batch_dim(tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
    # 단일 샘플 입력도 배치 차원을 갖도록 맞춰 네트워크 경로를 단순화한다.
    while tensor.ndim < target_ndim:
        tensor = tensor.unsqueeze(0)
    return tensor


class ActorHead(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        embed_size: int,
        padding_mask: bool = False,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.padding_mask = padding_mask
        self.device = device
        self.preprocess = preprocess_net
        self.layer_1 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )

    def forward(
        self, 
        obs: Any,
        state: Any = None,
        info: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Any]:
        mask = _to_tensor(_obs_get(obs, "mask"), self.device, torch.bool)
        mask = _ensure_batch_dim(mask, 2)

        # next_box 임베딩을 후보 임베딩과 매칭해 각 후보의 정책 로그릿을 계산한다.
        item_embedding, cand_embedding, hidden = self.preprocess(obs, state, mask)
        item_context = self.layer_1(item_embedding).mean(dim=1, keepdim=True)
        cand_embedding = self.layer_2(cand_embedding).transpose(1, 2)
        logits = torch.bmm(item_context, cand_embedding).squeeze(1)

        return logits, hidden
    

class CriticHead(nn.Module):
    def __init__(
        self,
        k_placement: int,
        preprocess_net: nn.Module,
        embed_size: int,
        padding_mask: bool = False,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.padding_mask = padding_mask
        self.device = device
        self.preprocess = preprocess_net
        self.k_placement = k_placement
        self.layer_1 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_3 = nn.Sequential(
            init_(nn.Linear(2 * embed_size, embed_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(embed_size, 1))
        )

    def forward(
        self, 
        obs: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        mask = _to_tensor(_obs_get(obs, "mask"), self.device, torch.bool)
        mask = _ensure_batch_dim(mask, 2)

        # 후보 임베딩은 유효 후보(mask)만 평균내서 상태 가치를 계산한다.
        item_embedding, cand_embedding, _ = self.preprocess(obs, None, mask)
        item_embedding = self.layer_1(item_embedding).mean(dim=1)
        cand_embedding = self.layer_2(cand_embedding)
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        cand_embedding = (cand_embedding * mask_f).sum(dim=1) / denom

        joint_embedding = torch.cat((item_embedding, cand_embedding), dim=-1)

        state_value = self.layer_3(joint_embedding)
        return state_value


class ShareNet(nn.Module):
    def __init__(
        self,
        k_placement: int = 100,
        box_max_size: int = 5,
        container_size: Sequence[int] = [10, 10, 10],
        embed_size: int = 32,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 6,
        dropout: float = 0,
        device: Union[str, int, torch.device] = "cpu",
        place_gen: str = "EMS",
    ) -> None:
        super().__init__()

        self.device = device
        self.k_placement = k_placement
        self.container_size = container_size
        self.place_gen = place_gen
        input_size = 7  # [box_dims(3), pos(3), orientation(1)]
        
        self.item_encoder = nn.Sequential(
            init_(nn.Linear(3, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embed_size)),
        )

        self.placement_encoder = nn.Sequential(
            init_(nn.Linear(input_size, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embed_size)),
        )
        
        self.backbone = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, 
        obs: Any,
        state: Any = None,
        mask: Union[np.ndarray, torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, Any]:
        boxes_array = _to_tensor(_obs_get(obs, "boxes_array"), self.device, torch.float32)
        next_box = _to_tensor(_obs_get(obs, "next_box"), self.device, torch.float32)
        bin_dims = _to_tensor(_obs_get(obs, "bin_dims"), self.device, torch.float32)
        cand_boxes = _to_tensor(_obs_get(obs, "candidate_boxes"), self.device, torch.float32)
        cand_pos = _to_tensor(_obs_get(obs, "candidate_positions"), self.device, torch.float32)
        cand_ori = _to_tensor(_obs_get(obs, "candidate_oris"), self.device, torch.float32)
        obs_mask = _to_tensor(_obs_get(obs, "mask"), self.device, torch.bool)

        boxes_array = _ensure_batch_dim(boxes_array, 3)
        next_box = _ensure_batch_dim(next_box, 3)
        bin_dims = _ensure_batch_dim(bin_dims, 2)
        cand_boxes = _ensure_batch_dim(cand_boxes, 3)
        cand_pos = _ensure_batch_dim(cand_pos, 3)
        cand_ori = _ensure_batch_dim(cand_ori, 2)
        obs_mask = _ensure_batch_dim(obs_mask, 2)

        if mask is None:
            mask_t = obs_mask
        else:
            mask_t = _to_tensor(mask, self.device, torch.bool)
            mask_t = _ensure_batch_dim(mask_t, 2)

        # mm 스케일 입력을 bin 크기 기준으로 정규화해 bin 타입이 달라도 수치 범위를 안정화한다.
        bw = bin_dims[:, 0:1].clamp_min(1.0)
        bd = bin_dims[:, 1:2].clamp_min(1.0)
        bh = bin_dims[:, 2:3].clamp_min(1.0)

        boxes_norm = boxes_array.clone()
        boxes_norm[:, :, 0] = boxes_norm[:, :, 0] / bw
        boxes_norm[:, :, 1] = boxes_norm[:, :, 1] / bd
        boxes_norm[:, :, 2] = boxes_norm[:, :, 2] / bh
        boxes_norm[:, :, 3] = boxes_norm[:, :, 3] / bw
        boxes_norm[:, :, 4] = boxes_norm[:, :, 4] / bd
        boxes_norm[:, :, 5] = boxes_norm[:, :, 5] / bh

        next_box_norm = next_box.clone()
        next_box_norm[:, :, 0] = next_box_norm[:, :, 0] / bw
        next_box_norm[:, :, 1] = next_box_norm[:, :, 1] / bd
        next_box_norm[:, :, 2] = next_box_norm[:, :, 2] / bh

        cand_boxes_norm = cand_boxes.clone()
        cand_boxes_norm[:, :, 0] = cand_boxes_norm[:, :, 0] / bw
        cand_boxes_norm[:, :, 1] = cand_boxes_norm[:, :, 1] / bd
        cand_boxes_norm[:, :, 2] = cand_boxes_norm[:, :, 2] / bh

        cand_pos_norm = cand_pos.clone()
        cand_pos_norm[:, :, 0] = cand_pos_norm[:, :, 0] / bw * 2.0 - 1.0
        cand_pos_norm[:, :, 1] = cand_pos_norm[:, :, 1] / bd * 2.0 - 1.0
        cand_pos_norm[:, :, 2] = cand_pos_norm[:, :, 2] / bh * 2.0 - 1.0

        # next_box(정방향/회전) 두 토큰을 item query로 사용한다.
        item_embedding = self.item_encoder(next_box_norm)

        cand_feature = torch.cat(
            [cand_boxes_norm, cand_pos_norm, cand_ori.unsqueeze(-1)],
            dim=-1,
        )
        placement_embedding = self.placement_encoder(cand_feature)

        for layer in self.backbone:
            item_embedding, placement_embedding = layer(item_embedding, placement_embedding, mask_t)

        return item_embedding, placement_embedding, state


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
