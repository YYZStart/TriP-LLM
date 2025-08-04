import numpy as np
import torch
import time
import torch.nn as nn
from functorch.einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModel
from Deepod.base_model import BaseDeepAD
from Deepod.utility import get_sub_seqs
from Util import set_llm_models
from layers.Attention import Encoder_TRSF, MultiHeadAttention
from torch.cuda.amp import autocast, GradScaler
from layers.Multi_Patch_Soft_Causal_Branch_Enconding import compute_sliding_windows_info, GlobalBranch, GateFusion, \
    MultiScaleAdapter

'''
This is the TriP_LLM implementation under DeepOD framework (https://github.com/xuhongzuo/DeepOD)
'''


class TriP_LLM(BaseDeepAD):

    def __init__(self, model_name="TriP_LLM", finetune=False, data_type=torch.float32,
                 use_amp=True, feature_dim=123, model_path="select_model", ablation="none",
                 seq_len=100, stride=1, pred_len=5,
                 patch_size=[16], patch_stride=4,
                 epochs=100, batch_size=128, lr=1e-3,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):

        super(TriP_LLM, self).__init__(
            model_name=model_name,
            data_type='ts', epochs=epochs, batch_size=batch_size, lr=lr,
            seq_len=seq_len, stride=stride,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.finetune = finetune
        self.data_type = data_type
        self.model_path = model_path
        self.pred_len = pred_len

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.ablation = ablation

        self.use_amp = use_amp

        if self.use_amp:
            self.scaler = GradScaler(enabled=True)
        else:
            self.scaler = None

        self.net = LLM_TAD(
            model_path=self.model_path,
            seq_len=self.seq_len,
            pred_len=self.pred_len,

            patch_size=self.patch_size,
            patch_stride=self.patch_stride,

            ablation=self.ablation,

            feature_dim=self.feature_dim,
            finetune=self.finetune,
            device=self.device,
            data_type=self.data_type
        ).to(self.device)

        print(f"now is using {self.model_path} LLM model")

    def fit(self, X, y=None):

        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)

        dataloader = DataLoader(seqs, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.net.train()

        with tqdm(total=self.epochs, desc="Training Progress", unit="epoch") as pbar:
            for epoch in range(self.epochs):
                t1 = time.time()

                loss = self.training(dataloader, epoch, self.epochs)

                pbar.set_postfix({
                    "loss": f"{loss:.6f}",
                    "time": f"{time.time() - t1:.1f}s"
                })
                pbar.update(1)

                print(f'epoch{epoch + 1:3d}, '
                      f'training loss: {loss:.6f}, '
                      f'time: {time.time() - t1:.1f}s')

        # self.decision_scores_ = self.decision_function(X)
        # self.labels_ = self._process_decision_scores()  # in base model

        return

    def decision_function(self, X, return_rep=False):
        """
        Computes the anomaly scores for the given data.
        """

        seqs = get_sub_seqs(X, seq_len=self.seq_len, stride=self.stride)

        dataloader = DataLoader(seqs, batch_size=self.batch_size, shuffle=False, drop_last=False)

        anomaly_scores = self.inference(dataloader)
        scores_final = np.mean(anomaly_scores, axis=(1, 2))  # (n,)

        print(f"scores_final shape: {scores_final.shape}")

        scores_final_pad = np.hstack([0 * np.ones(X.shape[0] - scores_final.shape[0]), scores_final])

        print(f"scores_final_pad shape: {scores_final_pad.shape}")

        return scores_final_pad

    def training(self, dataloader):
        """
        Conducts a training pass on the given DataLoader.
        """

        criterion = nn.MSELoss()
        train_loss = []

        for ii, batch_x in enumerate(dataloader):
            self.optimizer.zero_grad()
            batch_x = batch_x.to(dtype=self.data_type, device=self.device)  # (B, seq_len, feature_dim)

            if self.use_amp == True:

                with torch.cuda.amp.autocast():
                    outputs = self.net(batch_x)
                    loss = criterion(outputs, batch_x)

                # Scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:

                outputs = self.net(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                self.optimizer.step()

            train_loss.append(loss.item())

            if self.epoch_steps != -1 and ii > self.epoch_steps:
                break

        self.scheduler.step()

        return np.average(train_loss)

    def inference(self, dataloader):
        """
        Performs inference on the data provided by the DataLoader.
        """

        criterion = nn.MSELoss(reduction='none')

        score = []

        self.net.eval()

        with torch.no_grad():

            with tqdm(total=len(dataloader), desc="Inference Progress", unit="batch") as pbar:
                for batch_x in dataloader:
                    batch_x = batch_x.to(dtype=self.data_type, device=self.device)  # (B, seq_len, feature_dim)

                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.net(batch_x)
                            loss_ = criterion(outputs, batch_x)

                    else:
                        outputs = self.net(batch_x)
                        loss_ = criterion(outputs, batch_x)

                    loss_ = loss_.detach().cpu()

                    if loss_.dtype == torch.bfloat16:
                        loss_ = loss_.to(torch.float32)

                    loss_ = loss_.cpu().numpy()

                    score.append(loss_)

                    pbar.update(1)

        anomaly_scores = np.concatenate(score, axis=0)

        print(f"Score shape: {anomaly_scores.shape}")

        return anomaly_scores

    def save_pt_model(self, path: str):

        if self.net is None:
            print("[Warning] No network to save.")
            return
        ckpt = {"model_state_dict": self.net.state_dict()}
        torch.save(ckpt, path)

        if self.verbose:
            print(f"[save_model] checkpoint saved at: {path}")

    def load_pt_model(self, path: str, map_location="cuda"):
        """
        load checkpoint
        """

        ckpt = torch.load(path, map_location=map_location)
        state_dict = ckpt["model_state_dict"]

        if self.net is None:
            self.net = LLM_TAD(
                model_path=self.model_path,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                feature_dim=self.feature_dim,

                patch_size=self.patch_size,
                patch_stride=self.patch_stride,

                ablation=self.ablation,

                finetune=self.finetune,
                device=self.device,
                data_type=self.data_type
            ).to(self.device)

        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)

        print(f"missing={len(missing)}, unexpected={len(unexpected)}")

        if self.verbose:
            print(f"[load_pt_model] Model parameters loaded and state_dict restored from: {path}")

        return self

    def training_forward(self, batch_x, net, criterion):
        """define forward step in training"""
        return

    def inference_forward(self, batch_x, net, criterion):
        """define forward step in inference"""
        return

    def training_prepare(self, X, y):
        """define train_loader, net, and criterion"""
        return

    def inference_prepare(self, X):
        """define test_loader"""
        return


class LLM_TAD(nn.Module):

    def __init__(
            self,
            model_path: str = "gpt2",  # the selected LLM , you can refer Util.py
            seq_len: int = 10,
            feature_dim: int = 5,  # channel dimension M
            pred_len: int = 3,
            # number of prediction steps (no longer used since this is a reconstruction-approach model)

            patch_size: list[int] = [16],
            patch_stride: int = 4,

            out_dim: int = 128,
            ablation: str = "none",

            finetune: bool = False,  # finetune or not
            device: str = "cuda",
            data_type: torch.dtype = torch.bfloat16,

    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.pred_len = pred_len
        self.finetune = finetune

        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.out_dim = out_dim
        self.ablation = ablation

        self.data_type = data_type

        self.hidden_dim, select_model, attn_implementation = set_llm_models(model_path)

        try:
            self.gpt2_model = AutoModel.from_pretrained(
                select_model,
                torch_dtype=self.data_type,
                output_hidden_states=True,
                attn_implementation=attn_implementation,
                device_map="cuda"
            ).to(device)
        except Exception as e:
            print(f"Load {select_model} failed : {e}")

            print(f"Using AutoModelForCausalLM")

            self.gpt2_model = AutoModelForCausalLM.from_pretrained(
                select_model,
                return_dict_in_generate=True,
                output_hidden_states=True,
                torch_dtype=self.data_type,
                attn_implementation=attn_implementation,
                device_map="cuda"
            ).to(device)
        self.num_patches, self.patch_len = compute_sliding_windows_info(self.seq_len, min(self.patch_size),
                                                                        self.patch_stride)

        min_patch_size = min(self.patch_size)

        self.multi_patch_scaler = MultiScaleAdapter(
            patch_lens=self.patch_size,
            stride=self.patch_stride,
            in_channels=self.feature_dim,
            hidden_dim=128,
            out_dim=self.out_dim
        )

        self.global_branch = GlobalBranch(
            in_channels=self.feature_dim,
            conv_out=128,
            final_dim=self.out_dim,
            out_seq_len=self.num_patches,
            kernel_size=5,
        ).to(self.device, dtype=self.data_type)

        self.gate_fusion = GateFusion(
            dim_sliding=self.out_dim * self.feature_dim,
            dim_selection=self.out_dim * self.feature_dim,
            dim_global=self.out_dim,
            unify_dim=256,
            model_dim=self.hidden_dim,
        ).to(self.device, dtype=self.data_type)

        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, min_patch_size * self.feature_dim)
        )

        # 3) finetune or not
        if not finetune:
            for param in self.gpt2_model.parameters():
                param.requires_grad = False

        else:
            if model_path == "gpt2":
                for param in self.gpt2_model.parameters():
                    param.requires_grad = False
                for name, param in self.gpt2_model.named_parameters():

                    if "ln" in name:
                        param.requires_grad = True
            else:

                for param in self.gpt2_model.parameters():
                    param.requires_grad = False
                for name, param in self.gpt2_model.named_parameters():

                    if ("layernorm" in name) or ("model.norm" in name):
                        param.requires_grad = True

        if "removeLLM" in self.ablation:
            del self.gpt2_model

        if 'llm_to_trsf' in self.ablation:
            del self.gpt2_model
            self.basic_trsf = Encoder_TRSF(hidden_dim=self.hidden_dim)
            self.basic_trsf.to(self.device, dtype=self.data_type)

        if 'llm_to_attn' in self.ablation:
            del self.gpt2_model
            self.basic_attn = MultiHeadAttention(d_model=self.hidden_dim)
            self.basic_attn.to(self.device, dtype=self.data_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, M)
        return: (B, L, M) -> reconstruction
        """
        B, L, M = x.shape

        # 1) 多分支 + GateFusion => GPT2 => last_hid

        # multi_patch_fusion
        feat_sld, feat_sel, l_max = self.multi_patch_scaler(x)

        # Global Branch
        feat_glb = self.global_branch(x)

        fused_embed = self.gate_fusion(feat_sld, feat_sel, feat_glb)  # (B, num_patches, d_model)

        # if you want to test the ablations try here

        # if "none" in self.ablation:
        #
        #     llm_outputs = self.gpt2_model(inputs_embeds=fused_embed)
        #     last_hid = llm_outputs.hidden_states[-1]  # => [B, num_patches, d_model]
        # elif "removeLLM" in self.ablation:
        #
        #     last_hid = fused_embed
        # elif "llm_to_attn" in self.ablation:
        #
        #     attn_out, _ = self.basic_attn(fused_embed, fused_embed, fused_embed)
        #     last_hid = attn_out
        # elif "llm_to_trsf" in self.ablation:
        #
        #     trsf_out = self.basic_trsf(fused_embed)
        #     last_hid = trsf_out
        # else:
        #     raise ValueError(f"Unknown ablation mode: {self.ablation}")

        outputs = self.gpt2_model(inputs_embeds=fused_embed)
        last_hid = outputs.hidden_states[-1]  # (B, num_patches, d_model)

        # Patch Decoder: => (B, num_patches, patch_size, M)
        last_hid_2d = rearrange(last_hid, 'b np d -> (b np) d')
        decoded_patches_2d = self.patch_decoder(last_hid_2d)  # => (B*num_patches, patch_size*M)
        decoded_patches_4d = rearrange(
            decoded_patches_2d,
            '(b np) (p m) -> b np p m',
            b=B, np=self.num_patches,
            p=min(self.patch_size), m=self.feature_dim
        )

        # Overlap-Average of overlapped patches (B, l, p, M) to => (B, L, M)
        # Decode patch by patch and map each patch back to its original time positions.
        # For overlapping regions, take the average across patches.

        recon_sum = x.new_zeros(B, L, M)  # shape: (B, L, M), for value accumulation
        recon_count = x.new_zeros(B, L, 1)  # shape: (B, L, 1), for counting overlaps

        for i in range(self.num_patches):

            start = i * self.patch_stride
            end = start + min(self.patch_size)

            if end > L:
                end = L

            length = end - start
            if length <= 0:
                break  # Skip patches that are completely out of range

            patch_i_out = decoded_patches_4d[:, i, :length, :]

            recon_sum[:, start:end, :] += patch_i_out

            recon_count[:, start:end, 0] += 1

        # recon_sum now contains the summed decoded values for each time point
        # recon_count holds the number of patches that contributed to each time point

        # Perform overlap-average using broadcasting
        mask = (recon_count > 0)

        safe_count = recon_count.clone()

        safe_count[~mask] = 1

        safe_count_3d = safe_count.expand(-1, -1, M)

        recon_x = recon_sum / safe_count_3d

        mask_3d = mask.expand(-1, -1, M)
        recon_x[~mask_3d] = 0

        return recon_x  # shape (B, L, M)
