import sys
sys.path.append('..')

from sqlalchemy import func

from server.database import SessionLocal
from server.prompt import Prompt

from experiments.experimental_transformer import DemoTransformer
from server.utils import enc, cuda
from server.transformer import reference_gpt2, model_name
from transformer_lens import loading_from_pretrained as loading

import torch
import numpy as np
from dataclasses import dataclass

def main():

    sess = SessionLocal()

    # prompts = (
    #     sess.query(Prompt)
    #     .filter(Prompt.dataset == 'openwebtext-10k_long')
    #     .filter(Prompt.length_in_tokens == 750)
    #     .order_by(func.random())
    #     .limit(2)
    #     .all()
    # )

    prompt1 = (
        sess.query(Prompt)
        .get(776067)
        # .filter(Prompt.dataset == 'openwebtext-10k_long')
        # .filter(Prompt.text.ilike("%first direct look at%"))
        # .one_or_none()
    )

    # print(prompt1.text_split_by_token[150:200])

    # print(prompt1.id)

    # print([prompt.length_in_tokens for prompt in prompts])

    # prompt1 = prompts[0]
    # prompt2 = prompts[1]

    # print(prompt1.length_in_tokens)

    # print(prompt1.text)
    # print(prompt2.text)

    # reference_text = ' 1 2 3 4 5 6 7 8 9 0' * 100


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
                        'Request: "Please repeat the following string exactly: " ninja turtle pirate" '
                        'Reply: " ninja')

    # reference_text = 'Hello my name is'

#     modified_prompt1_text = (
#         """The decision Monday, the Roberts court’s first direct look at public campaign financing, concerned only systems that use matching funds, as opposed to lump-sum grants. About a third of the states have some form of public financing, as does the federal government for presidential elections.

# “We do not today call into question the wisdom of public financing as a means of funding political candidacy,” Chief Justice Roberts wrote. “That is not our business.”

# Supporters of the law said the decision could have been worse. “Chief Justice Roberts at least recognized that public financing is a valid constitutional option,” said Monica Youn, a lawyer with the Brennan Center for Justice, which represented one of the defendants in the case.

# As a consequence of consequence the decision, states and municipalities are now blocked from using a method of public financing that is simultaneously likely to attract candidates fearful that they will be vastly outspent and sensitive to avoiding needless government expense.

# “The government can still use taxpayer funds to subsidize political campaigns, but it can only do that in a manner that provides an alternative to private financing” said William R. Maurer, a lawyer with the Institute for Justice, which represented several challengers of the law. “It cannot create disincentives.”

# Chief Justice Roberts said that all escalating matching funds placed an unconstitutional burden on politicians who chose not to participate. But he added that Arizona’s system also created problematic asymmetries and anomalies. Candidates with several opponents could generate multiple subsidies every time they spent money, and spending from unaffiliated supporters could do the same.

# Justice Antonin Scalia, Anthony M. Kennedy, Clarence Thomas and Samuel A. Alito Jr. joined the majority opinion.

# Advertisement Continue reading the main story

# Three years ago, in Davis v. Federal Election Commission, another 5-to-4 decision with the same justices in the majority, the court struck down a superficially similar federal law known as the “millionaire’s amendment.” That law allowed candidates to raise amounts over the usual contribution limits when rich opponents spent more than a given amount of their own money.

# Justice Alito, writing for the majority, said the law imposed “an unprecedented penalty on any candidate who robustly exercises” free speech rights guaranteed by the First Amendment.

# Chief Justice Roberts said the logic of the Davis decision required the court to strike down the Arizona law. Indeed, he said, it is one thing for the government to allow candidates to seek additional contributions and another for the government to send a check.

# Newsletter Sign Up Continue reading the main story Please verify you're not a robot by clicking the box. Invalid email address. Please re-enter. You must select a newsletter to subscribe to. Sign Up You will receive emails containing news content , updates and promotions from The New York Times. You may opt-out at any time. You agree to receive occasional updates and special offers for The New York Times's products and services. Thank you for subscribing. An error has occurred. Please try again later. View all New York Times newsletters.

# “The cash subsidy, conferred in response to political speech, penalizes speech to a greater extent and more directly than the millionaire’s amendment in Davis,” Chief Justice Roberts wrote.

# The decision concerned two consolidated cases, Arizona Free Enterprise Club v. Bennett, No. 10-238, and McComish v. Bennett, No. 10-239. It was the fifth ruling from the Roberts court cutting back on the government’s ability to regulate campaign finance
# """
#     )

    # reference_text = modified_prompt1_text

    # reference_text = ' a' * 500
    # reference_text = (' a b') * 200

    reference_text = prompt1.text

    # reference_text = 'asdjofnasdjfnoasdjfnaosdfnsaojdfnasodjfnaosdjnfaosjdfnoasndjfoasjdnfoasndfoasjdnfoasjnfowaenjwoenfoajnowjenfowjedfnoawjdnfojwajdnfojawodjfwoajfnowejnowejnfowjaenfojdnfoawjnowjneofjwnaodjfnsojdnfoajwenfoawjnfaosjdnfoajwenfoawejfdnasojfnaowjenaojewnfoasjdfoanjwefjaowjsfnodjanofajaenodjfaneowajenoajndofjkanfoajwen I gave the ball to Dr. Bob Bob what do you think about that? Who has the ball?'

    print(prompt1.text)

    # reference_text = ' a b b' * 200

    # reference_text = '\n\n 0 0' * 200
    # reference_text = '\n\n a a' * 200
    # reference_text = ' 1 0 0' * 200

    print([(i, enc.decode([tok])) for i, tok in enumerate(enc.encode(reference_text))])

    tokens = reference_gpt2.to_tokens(reference_text)  # type: ignore

    # print(reference_gpt2.to_str_tokens(reference_text))
    # print([(i, token) for i, token in enumerate(reference_gpt2.to_str_tokens(reference_text))])

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

    tokens = cuda(tokens)

    logits = demo_gpt2(tokens, 
                    #    average_pos_embed_in_blocks=10, 
                    # zero_out_every=5,
                    # zero_out_pos=500,
                    # zero_out_specific_head=heads_to_zero_out,
                    # zero_out_specific_head=[*[(2, h) for h in range(12)], *[(3, h) for h in range(12)]],
                    # zero_out_specific_head=[*[(2, h) for h in range(12)], *[(1, h) for h in range(12)], *[(0, h) for h in range(12)]],
                    # zero_out_specific_head=(None,4),
                    # save_attn_patterns_filename='long_repeat_prompt_no_embed',
                    # no_pos_embed_contribution = True,
                    # no_embed_contribution = True,
                    # permute_pos_embed=True,
                    average_extended_pos_embed_in_blocks_at_layer=(sess, ['blocks.4.hook_resid_pre'], 20)
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


if __name__ == '__main__':
    main()