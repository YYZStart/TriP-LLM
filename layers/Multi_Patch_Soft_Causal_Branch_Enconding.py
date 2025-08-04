import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers.Causal_Convolution import CausalConv1d
from layers.ts_network_tcn import TCNnet


def compute_sliding_windows_info(L: int, patch_len: int, stride: int):
    num_patches = (L - patch_len) // stride + 1
    return num_patches, patch_len


# This Patching operation is adapted from https://github.com/thuml/Time-Series-Library
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, ):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return x, n_vars


class SlidingBranch(nn.Module):
    """

    # This is the Patching Branch in the TriP-LLM paper

    (B, L, M) -> (B, l, M*out_dim)
    """

    def __init__(self,
                 patch_len: int,
                 stride: int,
                 out_dim: int,
                 conv_out: int = 64,
                 kernel_size: int = 5):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.conv_out = conv_out
        self.kernel_size = kernel_size
        self.out_dim = out_dim

        # -- patching
        self.patch_embed = PatchEmbedding(patch_len, stride)

        # -- (Dilation conv 1,2)
        self.causal_conv = nn.Sequential(
            CausalConv1d(patch_len, conv_out,
                         kernel_size, dilation=1,
                         dropout=0.1, activation='SiLU'),
            CausalConv1d(conv_out, conv_out,
                         kernel_size, dilation=2,
                         dropout=0.1, activation='SiLU'),
        )

        # Depth-wise conv
        self.depthwise = nn.Conv1d(conv_out, conv_out,
                                   kernel_size=1, groups=conv_out)
        self.proj = nn.Linear(conv_out, out_dim)

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, M)
        return: (B, l, M*out_dim)
        """
        B, L, M = x.shape
        x = rearrange(x, 'b l m -> b m l')  # (B, M, L)

        patches_seq, nb_features = self.patch_embed(x)  # (B*M, l, p)
        bnm, l, p = patches_seq.shape

        y = rearrange(patches_seq, 'bnm l p -> bnm p l')  # (b*nm, p, l)
        y = self.causal_conv(y)
        y = self.depthwise(y)

        y = rearrange(y, 'bnm c l -> (bnm l) c')  # (b*nm*l, out_dim)
        y = self.proj(y)
        y = self.norm(y)
        y = rearrange(y, '(bnm l) d -> bnm l d', bnm=bnm, l=l)  # => (b*nm, l, out_dim)

        res = patches_seq.mean(dim=-1, keepdim=True)  # (b*nm, l, 1)
        res = res.expand(-1, -1, y.shape[-1])
        y = y + res

        # reshape to (B, l, M*out_dim)
        y = rearrange(y, '(b m) l d -> b l (m d)', b=B, m=M)
        return y


class SelectionBranch(nn.Module):
    def __init__(self,
                 in_channels: int,
                 patch_len: int,
                 stride: int,
                 hidden_dim: int,
                 out_dim: int):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_len, stride)
        self.sliding_conv = nn.Sequential(
            nn.Conv1d(out_dim, patch_len, 1),
            nn.ReLU(),
        )

        # score_net
        self.score_net = nn.Sequential(
            nn.Linear(patch_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # → (·,1)
        )

        self.tau = nn.Parameter(torch.tensor(0.0))

        # feature_net: projects the patch into out_dim
        self.feature_net = nn.Sequential(
            nn.Linear(patch_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.out_dim = out_dim
        self.in_channels = in_channels  # M

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, sliding_out: torch.Tensor) -> torch.Tensor:
        """
        x           : (B, L, M)
        return      : (B, l, M*out_dim)
        """
        B, L, M = x.shape
        series = rearrange(x, 'b l m -> b m l')  # (B,M,L)

        # patching
        patches_seq, _ = self.patch_embed(series)  # (B*M,l,p)
        bnm, l, p = patches_seq.shape  # bnm = B*M

        # ---- Align the shape of SlidingBranch features and add them to the patch tokens ----
        sld = rearrange(sliding_out, 'b l (m d) -> b m l d', m=M, d=self.out_dim)
        sld = self.sliding_conv(rearrange(sld, 'b m l d -> (b m) d l'))  # (B*M,p,l)
        sld = rearrange(sld, 'bnm p l -> bnm l p')  # (B*M,l,p)

        patch_tokens = patches_seq + sld  # (B*M,l,p)

        # scoring
        scores = self.score_net(rearrange(patch_tokens, 'bnm l p -> (bnm l) p'))
        scores = rearrange(scores, '(bnm l) 1 -> bnm l', l=l)  # (B*M,l)

        max_scores = scores.view(B, M, l).max(dim=1).values  # max-pool --> (B,l)
        avg_scores = scores.view(B, M, l).mean(dim=1)  # mean-pool --> (B,l)

        tau = torch.sigmoid(self.tau)

        scores_mix = tau * max_scores + (1 - tau) * avg_scores

        weights = F.softmax(scores_mix, dim=-1)  # (B,l)

        patch_feat = self.feature_net(rearrange(patch_tokens, 'bnm l p -> (bnm l) p'))
        patch_feat = rearrange(patch_feat, '(bnm l) d -> bnm l d', l=l)  # (B*M,l,d)

        #   broadcast weights back to (B*M,l,1)
        weights = weights.unsqueeze(1).expand(-1, M, -1)  # (B,M,l)
        weights = rearrange(weights, 'b m l -> (b m) l 1')  # (B*M,l,1)

        attn_feat = weights * patch_feat  # (B*M,l,d)

        # reshape
        feat_out = rearrange(attn_feat, '(b m) l d -> b l (m d)', b=B, m=M)  # (B,l,M*d)
        return feat_out


class MultiScaleAdapter(nn.Module):
    """
    Aggregate Multi-patches from Patching and Selecting Branches
    """

    def __init__(self,
                 patch_lens: list[int],
                 stride: int,
                 in_channels: int,
                 hidden_dim: int,
                 out_dim: int):
        super().__init__()

        patch_lens = sorted(patch_lens)

        self.scales = nn.ModuleList([
            nn.ModuleDict({
                'slide': SlidingBranch(p, stride, out_dim),
                'select': SelectionBranch(in_channels, p, stride,
                                          hidden_dim, out_dim)
            })
            for p in patch_lens
        ])
        self.patch_lens = patch_lens
        self.stride = stride

    @staticmethod
    def upsample_to_len(x: torch.Tensor, target_len: int):
        """
        1-D linear interpolation
        x: (B, l, D) → (B, target_len, D)
        """
        if x.size(1) == target_len:
            return x

        mode = 'nearest' if min(x.size(1), target_len) < 2 else 'linear'

        x_u = F.interpolate(
            rearrange(x, 'b l d -> b d l'),
            size=target_len,
            mode=mode,
            align_corners=False)
        return rearrange(x_u, 'b d l -> b l d')

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        slide_feats, select_feats, l_lens = [], [], []

        for mod, p_len in zip(self.scales, self.patch_lens):
            f_s = mod['slide'](x)  # (B,l_i,D)
            f_c = mod['select'](x, f_s)  # (B,l_i,D)
            slide_feats.append(f_s)
            select_feats.append(f_c)
            l_lens.append(f_s.size(1))  # l patch numbers

        l_max = max(l_lens)  # Align to the longest sequence, since shorter patches yield more patches

        slide_aligned = [self.upsample_to_len(f, l_max) for f in slide_feats]
        select_aligned = [self.upsample_to_len(f, l_max) for f in select_feats]

        scale_scores = torch.stack(
            [f.mean(dim=(1, 2)) for f in select_aligned], dim=-1)  # (B,S)
        sigma = F.softmax(scale_scores, dim=-1).unsqueeze(1).unsqueeze(-1)  # (B,1,S,1)

        stack_s = torch.stack(slide_aligned, dim=2)  # (B,l_max,S,D)
        stack_c = torch.stack(select_aligned, dim=2)  # (B,l_max,S,D)

        fused_s = (sigma * stack_s).sum(dim=2)  # (B,l_max,D)
        fused_c = (sigma * stack_c).sum(dim=2)  # (B,l_max,D)
        return fused_s, fused_c, l_max


class GlobalBranch(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_out: int,
                 final_dim: int,
                 out_seq_len: int,
                 kernel_size: int = 5):
        super().__init__()

        self.tcn = TCNnet(
            n_features=in_channels,
            n_hidden=[16, 32, 64],
            n_output=conv_out,
            kernel_size=kernel_size,
            dropout=0.1)

        self.linear = nn.Linear(conv_out, final_dim)

        self.pool = nn.AdaptiveMaxPool1d(out_seq_len)

    def forward(self, x: torch.Tensor):
        B, L, M = x.shape
        x = rearrange(x, 'b l m -> b m l')  # (B,M,L)
        y = self.tcn(x)  # (B,conv_out,L)
        y = F.relu(y)
        y = rearrange(y, 'b c l -> b l c')
        y = self.linear(y.view(-1, y.size(-1)))  # (B*L, final_dim)
        y = rearrange(y, '(b l) d -> b d l', b=B, l=L)
        y = self.pool(y)  # (B,final_dim,out_seq)
        y = rearrange(y, 'b d l -> b l d')
        return y  # (B,out_seq,final_dim)


class GateFusion(nn.Module):
    def __init__(self,
                 dim_sliding: int,
                 dim_selection: int,
                 dim_global: int,
                 unify_dim: int,
                 model_dim: int):
        super().__init__()

        self.unify_dim = unify_dim

        self.proj_sliding = nn.Linear(dim_sliding, unify_dim)
        self.proj_select = nn.Linear(dim_selection, unify_dim)
        self.proj_global = nn.Linear(dim_global, unify_dim)

        self.ln = nn.LayerNorm(unify_dim)

        self.drop = nn.Dropout(0.1)

        self.gate = nn.Linear(unify_dim * 3, 3)

        self.fusion_conv = nn.Conv1d(unify_dim, model_dim, kernel_size=1)

    def forward(self, f_s: torch.Tensor,
                f_c: torch.Tensor,
                f_g: torch.Tensor):
        s = self.proj_sliding(f_s)  # Sliding (Patching) Branch
        c = self.proj_select(f_c)  # Selecting ("C"hoosing) Branch
        g = self.proj_global(f_g)  # Global Branch

        s = self.ln(s)
        c = self.ln(c)
        g = self.ln(g)

        cat = torch.cat([s, c, g], dim=-1)  # (B,L,3*D)

        alpha_logits = self.drop(self.gate(cat))  # (B,L,3)

        alpha = F.softmax(alpha_logits, dim=-1)  # (B,L,3) across 3 branches
        alpha_s, alpha_c, alpha_g = alpha.unbind(dim=-1)
        alpha_s = alpha_s.unsqueeze(-1)  # (B,L,1)
        alpha_c = alpha_c.unsqueeze(-1)
        alpha_g = alpha_g.unsqueeze(-1)

        fuse = alpha_s * s + alpha_c * c + alpha_g * g  # (B,L,D)

        out = self.fusion_conv(
            rearrange(fuse, 'b l d -> b d l'))  # (B,dim_model,L)
        out = rearrange(out, 'b d l -> b l d')  # (B,L,dim_model)
        return out
