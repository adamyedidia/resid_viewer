import sys
sys.path.append('..')

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import StandardScaler

from server.database import SessionLocal
from server.direction import Direction, add_direction
from server.model import Model
from server.prompt import Prompt
from server.resid import Resid
from server.resid_writer import write_resids_for_prompt
from server.scaler import add_scaler
from server.transformer import reference_gpt2
sys.path.append('..')

import random
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

from server.utils import get_layer_num_from_resid_type, get_random_token, enc, has_unique_tokenization


def main():
    sess = SessionLocal()

    seq_length = 20

    starting_seq_length = random.randint(0, seq_length - 1)

    model = (
        sess.query(Model)
        .filter(Model.name == 'gpt2-small')
        .one_or_none()
    )

    keys = ['blocks.0.hook_attn_out', 'blocks.0.hook_mlp_out']


    for i in range(5000):

        token1 = get_random_token()
        token2 = get_random_token()

        print(i, enc.decode([token1]), enc.decode([token2]))

        token1_enc_prompt_long = [token1] * starting_seq_length + ([token2] * seq_length + [token1] * seq_length) * 30
        token2_enc_prompt_long = [token2] * starting_seq_length + ([token1] * seq_length + [token2] * seq_length) * 30

        token1_enc_prompt = token1_enc_prompt_long[:1023]
        token2_enc_prompt = token2_enc_prompt_long[:1023]

        token1_str_split_by_token = [enc.decode([token]) for token in token1_enc_prompt]
        token2_str_split_by_token = [enc.decode([token]) for token in token2_enc_prompt]

        token1_text = enc.decode(token1_enc_prompt)
        token2_text = enc.decode(token2_enc_prompt)

        if not has_unique_tokenization(token1_enc_prompt_long):
            continue

        if (existing_prompt := sess.query(Prompt).filter(Prompt.text == token1_text).first()) is not None:
            print(f'Prompt already exists: {existing_prompt}')
            continue

        dataset = f'automatic_dataset_{token1}_{token2}'

        prompt1 = Prompt(
            text=token1_text,
            encoded_text_split_by_token=token1_enc_prompt,
            text_split_by_token=token1_str_split_by_token,
            length_in_tokens=len(token1_enc_prompt),
            dataset=dataset,
        )

        prompt2 = Prompt(
            text=token2_text,
            encoded_text_split_by_token=token2_enc_prompt,
            text_split_by_token=token2_str_split_by_token,
            length_in_tokens=len(token2_enc_prompt),
            dataset=dataset,
        )

        sess.add(prompt1)
        sess.add(prompt2)

        sess.commit()

        write_resids_for_prompt(sess, prompt1, model, more_commits=True, verbose=True, 
                                keys=['blocks.0.hook_attn_out', 'blocks.0.hook_mlp_out'])

        for key in keys:
            layer_num = get_layer_num_from_resid_type(key)

            resids = (
                sess.query(Resid)
                .filter(Resid.model == model)
                .filter(Resid.dataset == dataset)
                .filter(Resid.layer == layer_num)
                .filter(Resid.type == key)
                .filter(Resid.head == None)
                .filter(Resid.token_position > 0)  # The leading |<endoftext>| token is weird
                .all()
            )

            X = np.array([resid.resid for resid in resids])
            print(X.shape)

            standard_scaler = StandardScaler()
            X_scaled = standard_scaler.fit_transform(X)

            # Perform PCA
            pca = PCA()
            X_pca = pca.fit_transform(X_scaled)

            print('PCA complete!')

            explained_variance_ratio = pca.explained_variance_ratio_

            generated_by_process = 'automatic_pca'

            scaler = add_scaler(
                sess,
                standard_scaler=standard_scaler,
                model=model,
                layer=layer_num,
                type=key,
                head=None,
                generated_by_process=generated_by_process,
                no_commit=True
            )

            for component_index, component in enumerate(pca.components_):
                direction = add_direction(
                    sess,
                    direction=component,
                    model=model,
                    layer=layer_num,
                    type=key,
                    head=None,
                    name=f'automatic_pca_{key}_{token1}_{token2}_{component_index}',
                    generated_by_process=generated_by_process,
                    component_index=component_index,
                    scaler=scaler,
                    fraction_of_variance_explained=explained_variance_ratio[component_index],
                    no_commit=True
                )
            
                if i < 10 and component_index == 0:
                    resids = (
                        sess.query(Resid)
                        .filter(Resid.model == model)
                        .filter(Resid.prompt == prompt1)
                        .filter(Resid.layer == 0)
                        .filter(Resid.head == None)
                        .filter(Resid.type == key)
                        .order_by(Resid.token_position.asc())
                        .all()
                    )

                    # dot_prods = []
                    # for resid in resids:
                    #     dot_prods.append(np.dot(np.array(resid.resid), np.array(direction.direction)))

                    # plt.plot(dot_prods)
                    # plt.show()

            sess.commit()


def parse_dataset_for_tokens(s):
    match = re.search(r'automatic_dataset_(\d+)_(\d+)', s)
    if match:
        x, y = map(int, match.groups())
        return int(x), int(y)
    else:
        return None, None


def get_best_solution():
    sess = SessionLocal()
    
    model = (
        sess.query(Model)
        .filter(Model.name == 'gpt2-small')
        .one_or_none()
    )

    prompts = (
        sess.query(Prompt)
        .filter(Prompt.dataset.startswith('automatic_dataset_'))
        .order_by(Prompt.created_at.desc())
        .all()
    )

    input_vectors = []
    output_vectors = []

    for i, prompt in enumerate(prompts):
        print(i, prompt)
        try:
            token1, token2 = parse_dataset_for_tokens(prompt.dataset)
        except Exception:
            token1, token2 = None, None

        if token1 is None or token2 is None:
            continue

        resids_for_prompt = (
            sess.query(Resid)
            .filter(Resid.model == model)            
            .filter(Resid.prompt == prompt)
            .filter(Resid.layer == 0)
            .filter(Resid.head == None)
            .filter(Resid.type == 'blocks.0.hook_attn_out')
            .order_by(Resid.token_position.asc())
            .all()
        )

        if len(resids_for_prompt) == 0:
            continue

        direction = (
            sess.query(Direction)
            .filter(Direction.model == model)
            .filter(Direction.layer == 0)
            .filter(Direction.head == None)
            .filter(Direction.type == 'blocks.0.hook_attn_out')
            .filter(Direction.name == f'automatic_pca_blocks.0.hook_attn_out_{token1}_{token2}_0')
            .filter(Direction.component_index == 0)
            .one_or_none()
        )

        if direction is None:
            continue

        token1_dot_products = []
        token2_dot_products = []
        last_5_tokens = []
        for resid in resids_for_prompt[:100]:

            dot_prod = np.dot(np.array(resid.resid), np.array(direction.direction))
            last_5_tokens.append(resid.encoded_token)
            last_5_tokens = last_5_tokens[-5:]
            if all([t == token1 for t in last_5_tokens]):
                token1_dot_products.append(dot_prod)
            elif all([t == token2 for t in last_5_tokens]):
                token2_dot_products.append(dot_prod)

        if len(token1_dot_products) == 0 or len(token2_dot_products) == 0:
            raise Exception('Something weird happened')
        
        if max(token1_dot_products) < min(token2_dot_products):
            # Then this is token2 - token1
            input_vectors.append((reference_gpt2.W_E[token2] - reference_gpt2.W_E[token1]).detach().numpy())
            output_vectors.append(direction.direction)

            # from scipy.stats.stats import pearsonr   
            # print(pearsonr(np.dot((reference_gpt2.W_E[token2] - reference_gpt2.W_E[token1]).detach().numpy(), pickle.load(open('pickle_files/learned_mat.p', 'rb'))), direction.direction))


            # plt.plot(np.dot((reference_gpt2.W_E[token2] - reference_gpt2.W_E[token1]).detach().numpy(), pickle.load(open('pickle_files/learned_mat.p', 'rb'))), label='learned')
            # plt.plot(direction.direction, label='direction')
            # plt.legend()
            # plt.show()

        elif max(token2_dot_products) < min(token1_dot_products):
            # Then this is token1 - token2
            input_vectors.append((reference_gpt2.W_E[token1] - reference_gpt2.W_E[token2]).detach().numpy())
            output_vectors.append(direction.direction)

            # from scipy.stats.stats import pearsonr   
            # print(pearsonr(np.dot((reference_gpt2.W_E[token2] - reference_gpt2.W_E[token1]).detach().numpy(), pickle.load(open('pickle_files/learned_mat.p', 'rb'))), direction.direction))


            # plt.plot(np.dot((reference_gpt2.W_E[token2] - reference_gpt2.W_E[token1]).detach().numpy(), pickle.load(open('pickle_files/learned_mat.p', 'rb'))), label='learned')
            # plt.plot(direction.direction, label='direction')
            # plt.legend()
            # plt.show()

        else:
            print('Weird behavior on prompt', prompt)
            dot_prods = []
            for resid in resids_for_prompt:
                dot_prods.append(np.dot(np.array(resid.resid), np.array(direction.direction)))

            plt.plot(dot_prods)
            plt.show()

            continue
            # raise Exception('Something weird happened')
        
    input_vectors = np.array(input_vectors)
    output_vectors = np.array(output_vectors)

    print(input_vectors.shape)
    print(output_vectors.shape)

    learned_mat = np.dot(np.linalg.pinv(input_vectors), output_vectors)

    pickle.dump(learned_mat, open('pickle_files/learned_mat.p', 'wb'))

    print(learned_mat.shape)
    plt.matshow(learned_mat)
    plt.show()


def try_learned_mat():
    sess = SessionLocal()

    learned_mat = pickle.load(open('pickle_files/learned_mat.p', 'rb'))
    prompts = sess.query(Prompt).order_by(Prompt.created_at.desc()).all()


    model = (
        sess.query(Model)
        .filter(Model.name == 'gpt2-small')
        .one_or_none()
    )

    for prompt in prompts:

        resids = (
            sess.query(Resid)
            .filter(Resid.model == model)
            .filter(Resid.prompt == prompt)
            .filter(Resid.layer == 0)
            .filter(Resid.head == None)
            .filter(Resid.type == 'blocks.0.hook_attn_out')
            .order_by(Resid.token_position.asc())
            .all()
        )

        if not resids:
            continue

        print(prompt)
        print(model)
        print(resids)

        last_5_tokens = []

        for resid in resids:

            last_5_tokens.append(resid.encoded_token)
            last_5_tokens = last_5_tokens[-5:]
            if all([t == resid.encoded_token for t in last_5_tokens]):
                plt.plot(np.dot(learned_mat, reference_gpt2.W_E[resid.encoded_token].detach().numpy()))
                plt.plot(resid.resid)
                plt.show()



if __name__ == "__main__":
    main()
    # get_best_solution()
    # try_learned_mat()

