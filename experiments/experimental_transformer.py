import sys
sys.path.append('..')

import torch.nn as nn
import torch
import einops
from fancy_einsum import einsum
from transformer_lens import HookedTransformer
from transformer_lens import loading_from_pretrained as loading
from transformer_lens.utils import gelu_new
from server.utils import M1_MAC, cuda
from dataclasses import dataclass
import math
import pickle

model_name = "gpt2-small"
# model_name = "pythia-70m"

reference_gpt2 = HookedTransformer.from_pretrained(model_name, fold_ln=False, center_unembed=False,
                                                   center_writing_weights=False)

models_dict = {'gpt2-small': reference_gpt2}

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])  # type: ignore

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)

tokens = cuda(tokens)
print(tokens)
logits, cache = reference_gpt2.run_with_cache(tokens)

log_probs = logits.log_softmax(dim=-1)  # type: ignore
probs = logits.log_softmax(dim=-1)  # type: ignore

next_token = logits[0, -1].argmax(dim=-1)  # type: ignore

next_tokens = torch.cat(
    [tokens, torch.tensor(next_token, device='cpu' if M1_MAC else 'cuda', dtype=torch.int64)[None, None]], dim=-1)
new_logits = reference_gpt2(next_tokens)

for activation_name, activation in cache.cache_dict.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(activation_name, activation.shape)

for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(name, param.shape)


@dataclass
class Config:
    d_model: int = 768
    debug: bool = False
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


# Returns the configuration parameters of the model as a basic Config dataclass
def get_basic_config(model_name: str, **kwargs) -> Config:
    return Config(
        **{k: v for k, v in loading.get_pretrained_model_config(model_name,
                                                                **kwargs).to_dict().items() if k in [
               'd_model',
               'layer_norm_eps',
               'd_vocab',
               'init_range',
               'n_ctx',
               'd_head',
               'd_mlp',
               'n_heads',
               'n_layers',
           ]})


cfg = get_basic_config(model_name)


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        if self.cfg.debug: print("Residual:", residual.shape)
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1",
                               "mean") + cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized


# _ = rand_float_test(LayerNorm, [2, 4, 768])
# _ = load_gpt2_test(LayerNorm, reference_gpt2.ln_final, "blocks.11.hook_resid_post")

class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :]  # [batch, position, d_model]
        # visualize_tensor(self.W_E, 'WE')
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed


# rand_int_test(Embed, [2, 4])
# load_gpt2_test(Embed, reference_gpt2.embed, tokens)

class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1), :]  # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug: print("pos_embed:", pos_embed.shape)
        return pos_embed


# rand_int_test(PosEmbed, [2, 4])
# load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
class Attention(nn.Module):
    def __init__(self, cfg, layer_num):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))

        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))
        self.layer_num = layer_num

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cpu" if M1_MAC else "cuda"))

    def forward(self, normalized_resid_pre, zero_out_pos=None, save_attn_patterns_filename=None,
                zero_out_specific_head=None):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)

        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head",
                   normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
                   normalized_resid_pre, self.W_K) + self.b_K

        attn_scores = einsum(
            "batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)

        if zero_out_specific_head is not None:
            # print('zeroing', head_number, layer_number)
            attn_scores = self.apply_causal_mask_specific_head_ablated(attn_scores, 
                                                                       zero_out_specific_heads=zero_out_specific_head,
                                                                       zero_out_pos=zero_out_pos)

        else:
            attn_scores = self.apply_causal_mask(attn_scores, zero_out_pos=zero_out_pos)

        # print(zero_out_pos)

        # if zero_out_pos is not None:
        #     attn_scores = self.apply_zero_out_single_attend(attn_scores, zero_out_pos)

        pattern = attn_scores.softmax(dim=-1)  # [batch, n_head, query_pos, key_pos]


        for i in range(12):
            if save_attn_patterns_filename:
                pickle.dump(pattern[0][i].detach().numpy(), 
                            open(f'pickle_files/{save_attn_patterns_filename}_L{self.layer_num}_H{i}.p', 'wb'))
            # import matplotlib.pyplot as plt
            # plt.matshow(pattern[0][i].detach().numpy(), vmin=0, vmax=0.1)
            # plt.colorbar()
            # plt.show()

        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
                   normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head",
                   pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z,
                          self.W_O) + self.b_O

        return attn_out

    def apply_causal_mask(self, attn_scores, zero_out_pos=None):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device),
                          diagonal=1).bool()
        
        if zero_out_pos is not None:
            mask[zero_out_pos][zero_out_pos-1] = 1
        # import matplotlib.pyplot as plt
        # plt.matshow(mask.detach().numpy())
        # plt.show()        
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

    def apply_causal_mask_specific_head_ablated(self, attn_scores, zero_out_specific_heads, zero_out_pos=None):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask_2d = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), 
                             diagonal=1).bool()
                             
        # Extend the 2D mask to 4D mask: [batch, n_heads, query_pos, key_pos]
        mask_4d = mask_2d.repeat(attn_scores.size(0), 
                                 attn_scores.size(1), 
                                 1, 
                                 1)

        # print(zero_out_pos is not None and (layer_numbers is None or self.layer_num in layer_numbers))
        # print(self.layer_num)
        # print(head_numbers)

        if zero_out_pos is not None:
            for layer_num, head_num in zero_out_specific_heads:
                if head_num is None and layer_num == self.layer_num:
                    print(f'Zeroing out layer {self.layer_num}!')
                    mask_4d[0, :, zero_out_pos, zero_out_pos-1] = 1
                else:
                    if layer_num == self.layer_num:
                        mask_4d[0, head_num, zero_out_pos, zero_out_pos-1] = 1

        # import matplotlib.pyplot as plt
        # for i in range(12):
        #     plt.matshow(mask_4d[0][i].detach().numpy())
        #     plt.show()

        attn_scores.masked_fill_(mask_4d, self.IGNORE)
        return attn_scores

    # def apply_zero_out_single_attend(self, attn_scores, pos):
    #     mask = torch.zeros(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
    #     mask[pos][pos-1] = 1

    #     attn_scores.masked_fill_(mask, self.IGNORE)
    #     return attn_scores

# rand_float_test(Attention, [2, 4, 768])
# load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["blocks.0.ln1.hook_normalized"])

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch position d_model, d_model d_mlp -> batch position d_mlp", normalized_resid_mid,
                     self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch position d_mlp, d_mlp d_model -> batch position d_model", post, self.W_out) + self.b_out
        return mlp_out


# rand_float_test(MLP, [2, 4, 768])
# load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["blocks.0.ln2.hook_normalized"])

class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_num):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg, layer_num)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre, zero_out_pos=None, save_attn_patterns_filename=False, 
                zero_out_specific_head=None):
        # resid_pre [batch, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre, zero_out_pos=zero_out_pos, 
                             save_attn_patterns_filename=save_attn_patterns_filename,
                             zero_out_specific_head=zero_out_specific_head)
        resid_mid = resid_pre + attn_out

        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post


# rand_float_test(TransformerBlock, [2, 4, 768])
# load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
        logits = einsum("batch position d_model, d_model d_vocab -> batch position d_vocab", normalized_resid_final,
                        self.W_U) + self.b_U
        return logits


class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, layer_num) for layer_num in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens, average_pos_embed=False, zero_out_pos=None, 
                save_attn_patterns_filename=False, no_pos_embed_contribution=False,
                no_embed_contribution=False,
                zero_out_specific_head=None):
        # tokens [batch, position]
        embed = self.embed(tokens)
        # visualize_tensor(self.embed.W_E, 'we')
        # print(embed.shape)      
        # visualize_tensor(embed, "Embedding")  
        pos_embed = self.pos_embed(tokens)

        print(pos_embed.shape)

        if average_pos_embed:

            first_part = pos_embed[:, :-10, :]
            last_part = pos_embed[:, -10:, :]

            mean_of_last_10 = last_part.mean(dim=1).unsqueeze(1)

            duplicated_mean = mean_of_last_10.repeat_interleave(last_part.shape[1], dim=1)

            # print(first_part.shape)
            # print(mean_of_last_10.shape)

            pos_embed = torch.cat((first_part, duplicated_mean), dim=1)

        # print(pos_embed.shape)
        # visualize_tensor(pos_embed, "Positional Embedding")
        if no_pos_embed_contribution:
            residual = embed
        elif no_embed_contribution:
            residual = pos_embed
        else:
            residual = embed + pos_embed
        # print(residual.shape)
        # visualize_tensor(residual, "Residual")
        for block in self.blocks:
            residual = block(residual, zero_out_pos=zero_out_pos,
                             save_attn_patterns_filename=save_attn_patterns_filename,
                             zero_out_specific_head=zero_out_specific_head)
            # print(residual)
        normalized_resid_final = self.ln_final(residual)
        # print(normalized_resid_final)
        logits = self.unembed(normalized_resid_final)
        # print(logits)
        # logits have shape [batch, position, logits]
        return logits
