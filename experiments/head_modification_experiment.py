import sys
sys.path.append('..')

M1_MAC = True
import torch
import numpy as np
from experiments.experimental_transformer import DemoTransformer
from server.transformer import reference_gpt2, model_name
from server.utils import enc
from dataclasses import dataclass
from transformer_lens import loading_from_pretrained as loading
import pickle
import matplotlib.pyplot as plt

SAVE_MATRICES = False
COMPARE_MATRICES = True

if SAVE_MATRICES:

    reference_text = ('Request: Please repeat the following string exactly: "hello" '
                        'Reply: "hello". '
                        'Request: "Please repeat the following string exactly: "gorapopefm" '
                        'Reply: "gorapopefm" '
                        'Request: "Please repeat the following string exactly: "adfgpinaie" '
                        'Reply: "adfgpinaie" '
                        'Request: "Please repeat the following string exactly: " poaspdmfpm" '
                        'Reply: " poaspdmfpm" '
                        'Request: "Please repeat the following string exactly: "wplmedpmdp" '
                        'Reply: "wplmedpmdp" '
                        'Request: "Please repeat the following string exactly: "pvofampovm" '
                        'Reply: "pvofampovm" '
                        'Request: "Please repeat the following string exactly: "poemfvpoe" '
                        'Reply: "poemfvpoe" '
                        'Request: "Please repeat the following string exactly: "vfavn" '
                        'Reply: "vfavn" '
                        'Request: "Please repeat the following string exactly: "sopqmx" '
                        'Reply: "sopqmx" '
                        'Request: "Please repeat the following string exactly: "france" '
                        'Reply: "france" '
                        'Request: "Please repeat the following string exactly: "vilion" '
                        'Reply: "pvofampovm" ' 
                        'Request: "Please repeat the following string exactly: " jack" '
                        'Reply: " jack" '
                        'Request: "Please repeat the following string exactly: "aervaxv" '
                        'Reply: "aervaxv" '
                        'Request: "Please repeat the following string exactly: " poem" '
                        'Reply: " poem" '
                        'Request: "Please repeat the following string exactly: " Reddit" '
                        'Reply: " Reddit" '
                        'Request: "Please repeat the following string exactly: "irnnrf" '
                        'Reply: "irnnrf" '
                        'Request: "Please repeat the following string exactly: "wepoc" '
                        'Reply: "wepoc" '
                        'Request: "Please repeat the following string exactly: "propmfpm" '
                        'Reply: "propmfpm" '
                        'Request: "Please repeat the following string exactly: " Germany" '
                        'Reply: " Germany""'
                        'Request: "Please repeat the following string exactly: "rathasoadga" '
                        'Reply: "rathasoadga" '
                        'Request: "Please repeat the following string exactly: "1pdjpm3efe4" '
                        'Reply: "1pdjpm3efe4" '
                        'Request: "Please repeat the following string exactly: "apple orange" '
                        'Reply: "')

    # reference_text = 'Hello my name is'

    tokens = reference_gpt2.to_tokens(reference_text)

    print(reference_gpt2.to_str_tokens(reference_text))
    print([(i, token) for i, token in enumerate(reference_gpt2.to_str_tokens(reference_text))])

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

    def get_basic_config(model_name: str, **kwargs):
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


    demo_gpt2 = DemoTransformer(get_basic_config(model_name=model_name))
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

    def cuda(x):
        return x.to('cpu') if M1_MAC else x.cuda()

    tokens = cuda(tokens)
    logits = demo_gpt2(tokens, 
                    #    average_pos_embed=True, 
                    #    zero_out_pos=500,
                    save_attn_patterns_filename='long_repeat_prompt_no_embed',
                    # no_pos_embed_contribution = True,
                    no_embed_contribution = True,
                    )

    last_logits = logits[-1, -1]  # type: ignore
    # Apply softmax to convert the logits to probabilities
    probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()

    print(probabilities)

    # Get the indices of the top 10 probabilities
    topk_indices = np.argpartition(probabilities, -10)[-10:]
    # Get the top 10 probabilities
    topk_probabilities = probabilities[topk_indices]
    # Get the top 10 tokens
    topk_tokens = [enc.decode([i]) for i in topk_indices]

    # Print the top 10 tokens and their probabilities
    for token, probability in zip(topk_tokens, topk_probabilities):
        print(f"Token: {token}, Probability: {probability}")

if COMPARE_MATRICES:
    for i in range(12):
        for j in range(12):
            mat1 = pickle.load(open(f'pickle_files/long_repeat_prompt_L{i}_H{j}.p', 'rb'))
            mat2 = pickle.load(open(f'pickle_files/long_repeat_prompt_no_pos_embed_L{i}_H{j}.p', 'rb'))
            mat3 = pickle.load(open(f'pickle_files/long_repeat_prompt_no_embed_L{i}_H{j}.p', 'rb'))

            print(i, j)

            print(f'mat1 norm', np.linalg.norm(mat1))
            print(f'mat2 norm', np.linalg.norm(mat2))
            print(f'mat3 norm', np.linalg.norm(mat3))

            plt.matshow(mat1, vmin=0, vmax=0.1)
            plt.colorbar()
            plt.show()

            plt.matshow(mat2, vmin=0, vmax=0.1)
            plt.colorbar()
            plt.show()


            plt.matshow(mat3, vmin=0, vmax=0.1)
            plt.colorbar()
            plt.show()

        break
