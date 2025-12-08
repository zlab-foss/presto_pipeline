import math
import torch
import torch.nn as nn
from einops import repeat

from layers import (BANDS_GROUPS_IDX, NUM_DYNAMIC_WORLD_CLASSES,
                    get_sinusoid_encoding_table, month_to_tensor, get_month_encoding_table, Block)

from typing import  Union, Optional, Tuple



class Encoder(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        channel_embed_ratio: float = 0.25,
        month_embed_ratio: float = 0.25,
        depth=2,
        mlp_ratio=4,
        num_heads=8,
        max_sequence_length=24,
    ):
        super().__init__()

        self.band_groups = BANDS_GROUPS_IDX
        self.embedding_size = embedding_size

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx for idx, (group_name, _) in enumerate(self.band_groups.items())
        }
        self.band_group_to_idx["dynamic_world"] = max(self.band_group_to_idx.values()) + 1

        self.eo_patch_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(group), embedding_size)
                for group_name, group in self.band_groups.items()
            }
        )
        self.dw_embed = nn.Embedding(
            num_embeddings=NUM_DYNAMIC_WORLD_CLASSES + 1, embedding_dim=embedding_size
        )
        self.latlon_embed = nn.Linear(3, embedding_size)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)

        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        pos_embedding_size = int(embedding_size * (1 - (channel_embed_ratio + month_embed_ratio)))
        channel_embedding_size = int(embedding_size * channel_embed_ratio)
        month_embedding_size = int(embedding_size * month_embed_ratio)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, pos_embedding_size), requires_grad=False
        )
        month_tab = get_month_encoding_table(month_embedding_size)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        self.channel_embed = nn.Embedding(
            num_embeddings=len(self.band_groups) + 1, embedding_dim=channel_embedding_size
        )

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def cartesian(latlons: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # an embedding is calculated for all timesteps. This is then expanded
            # for each timestep in the sequence
            latlon_radians = latlons * math.pi / 180
            lats, lons = latlon_radians[:, 0], latlon_radians[:, 1]
            x = torch.cos(lats) * torch.cos(lons)
            y = torch.cos(lats) * torch.sin(lons)
            z = torch.sin(lats)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def mask_tokens(x, mask):
        summed = mask.sum(
            dim=(1, 2)
        )  # summed tells me the number of masked elements per batch idx
        assert summed.max() == summed.min(), f"{summed.max()}, {summed.min()}"

        batch_size = x.shape[0]
        removed_elements_per_batch = int(summed.max() / mask.shape[2])
        kept_elements_per_batch = x.shape[1] - removed_elements_per_batch
        embedding_dim = x.shape[-1]

        # we want the mask to just be the indices of the masked tokens
        indices = repeat(torch.arange(0, x.shape[1]).long().to(x.device), "d -> b d", b=x.shape[0])

        x = x[~mask.bool()].view(batch_size, kept_elements_per_batch, embedding_dim)

        mask = mask[:, :, 0]
        kept_indices = indices[~mask.bool()].view(batch_size, kept_elements_per_batch)
        removed_indices = indices[mask.bool()].view(batch_size, removed_elements_per_batch)

        return x, kept_indices, removed_indices

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        month: Union[torch.Tensor, int] = 0,
        eval_task: bool = True,
    ):
        device = x.device

        if mask is None:
            mask = torch.zeros_like(x, device=x.device).float()

        months = month_to_tensor(month, x.shape[0], x.shape[1], device)
        month_embedding = self.month_embed(months)
        positional_embedding = repeat(
            self.pos_embed[:, : x.shape[1], :], "b t d -> (repeat b) t d", repeat=x.shape[0]
        )

        # we assume the number of masked patches is the same
        # for all items in the batch. Otherwise things become a headache
        all_tokens, all_masks = [], []

        for channel_group, channel_idxs in self.band_groups.items():
            tokens = self.eo_patch_embed[channel_group](x[:, :, channel_idxs])
            channel_embedding = self.channel_embed(
                torch.tensor(self.band_group_to_idx[channel_group]).long().to(device)
            )
            channel_embedding = repeat(channel_embedding, "d -> b t d", b=x.shape[0], t=x.shape[1])
            if channel_group == "SRTM":
                # for SRTM, we reduce it to a single token instead of
                # a token per timestep
                channel_wise_positional_embedding = torch.cat(
                    (
                        torch.zeros_like(month_embedding[:, 0:1]),
                        channel_embedding[:, 0:1],
                        torch.zeros_like(positional_embedding[:, 0:1]),
                    ),
                    dim=-1,
                )
                indices = slice(0, 1)
            else:
                channel_wise_positional_embedding = torch.cat(
                    (month_embedding, channel_embedding, positional_embedding), dim=-1
                )
                indices = slice(None)

            tokens = tokens[:, indices]
            tokens += channel_wise_positional_embedding
            all_tokens.append(tokens)
            group_mask = repeat(
                torch.max(mask[:, indices, channel_idxs], dim=-1)[0],
                "b t -> b t d",
                d=tokens.shape[-1],
            )
            all_masks.append(group_mask)

        # then, dynamic world
        tokens = self.dw_embed(dynamic_world)
        channel_embedding = self.channel_embed(
            torch.tensor(self.band_group_to_idx["dynamic_world"]).long().to(device)
        )
        channel_embedding = repeat(channel_embedding, "d -> b t d", b=x.shape[0], t=x.shape[1])
        positional_embedding = torch.cat(
            (month_embedding, channel_embedding, positional_embedding), dim=-1
        )
        tokens += positional_embedding
        all_tokens.append(tokens)

        # now we calculate the mask for these [b, t] tokens
        group_mask = repeat(
            dynamic_world == NUM_DYNAMIC_WORLD_CLASSES, "b t -> b t d", d=tokens.shape[-1]
        )
        all_masks.append(group_mask)

        x = torch.cat(all_tokens, dim=1)  # [batch, timesteps, embedding_dim]
        mask = torch.cat(all_masks, dim=1)  # [batch, timesteps, embedding_dim]

        x, kept_indices, removed_indices = self.mask_tokens(x, mask)

        # append latlon tokens
        latlon_tokens = self.latlon_embed(self.cartesian(latlons)).unsqueeze(1)
        x = torch.cat((latlon_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # mask will be a boolean of shape [batch, total_num_tokens]
        if eval_task:
            return self.norm(x.mean(dim=1))
        return self.norm(x), kept_indices, removed_indices


class _MLPHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_layer_sizes: Tuple[int, ...] = (256,),
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        act = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
        }.get(activation.lower(), nn.GELU)

        layers = []
        d = in_features
        for h in hidden_layer_sizes:
            layers += [nn.Linear(d, h), act()]
            if dropout > 0:
                layers += [nn.Dropout(p=dropout)]
            d = h
        layers += [nn.Linear(d, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)