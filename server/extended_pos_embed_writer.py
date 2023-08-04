import sys

sys.path.append('..')

import random
from server.database import SessionLocal
from server.extended_pos_embed import ExtendedPosEmbed
from server.model import Model
from server.prompt import Prompt
from server.resid_writer import write_resids_for_prompt
from server.resid import Resid

from server.utils import has_unique_tokenization, enc, mean_subtract

import numpy as np
import matplotlib.pyplot as plt


def main():
    NUM_RANDOM_TOKENS_TO_AVERAGE_OVER = 100

    sess = SessionLocal()

    token_counter = 0

    prompts: list[Prompt] = []

    model = sess.query(Model).filter(Model.name == 'gpt2-small').first()

    if not model:
        raise Exception('No model found')

    while True:
        random_token = random.randint(0, 50255)

        encoded_prompt_text = [random_token] * 1023
        prompt_text = enc.decode(encoded_prompt_text)

        if has_unique_tokenization([random_token] * 1023):
            print(f'Writing {enc.decode([random_token])}')

            prompt = Prompt(
                text=prompt_text,
                dataset='random_repeated_tokens',
                encoded_text_split_by_token=encoded_prompt_text,
                text_split_by_token=[enc.decode([t]) for t in encoded_prompt_text],
                length_in_tokens=len(encoded_prompt_text),
            )

            prompts.append(prompt)
            sess.add(prompt)
            sess.commit()

            write_resids_for_prompt(sess, prompt, model,
                                    keys=[*[f'blocks.{i}.ln1.hook_normalized' for i in range(12)], 
                                          *[f'blocks.{i}.hook_resid_pre' for i in range(12)]])

            token_counter += 1

        print(token_counter)
        if token_counter > NUM_RANDOM_TOKENS_TO_AVERAGE_OVER:
            break


def write_extended_pos_embed():
    sess = SessionLocal()

    prompts = (
        sess.query(Prompt)
        .filter(Prompt.dataset == 'random_repeated_tokens')
        .order_by(Prompt.created_at.desc())
        .all()
    )

    model = sess.query(Model).filter(Model.name == 'gpt2-small').first()

    if not model:
        raise Exception('No model found')


    for layer_num in range(12):
        for key in [f'blocks.{layer_num}.ln1.hook_normalized', f'blocks.{layer_num}.hook_resid_pre']:
            print(key)

            matrices = []
            tokens = []

            for prompt in prompts:
                resids = (
                    sess.query(Resid)
                    .filter(Resid.model == model)
                    .filter(Resid.prompt == prompt)
                    .filter(Resid.dataset == 'random_repeated_tokens')
                    .filter(Resid.layer == layer_num)
                    .filter(Resid.type == key)
                    .filter(Resid.head == None)
                    .order_by(Resid.token_position)  # The leading |<endoftext>| token is weird
                    .all()
                )
                
                prompt_token = resids[0].prompt.text_split_by_token[1]

                tokens.append(prompt_token)

                matrix = np.array([resid.resid for resid in resids])

                matrix = mean_subtract(matrix)
                matrices.append(matrix)

            average_matrix = sum(matrices, np.zeros(matrices[0].shape)) / len(matrices)  # type: ignore
            tokens_used = {token: None for token in tokens}
            hashed_tokens_used = hash(''.join(tokens))

            for i, row in enumerate(average_matrix):
                sess.add(ExtendedPosEmbed(
                    model=model,
                    layer=layer_num,
                    extended_pos_embed=row,
                    type=key,
                    token_position=i,
                    tokens_used_to_compute=tokens_used,
                    tokens_used_to_compute_hash=str(hashed_tokens_used),
                ))

            sess.commit()



if __name__ == '__main__':
    # main()
    write_extended_pos_embed()