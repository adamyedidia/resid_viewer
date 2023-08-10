
import sys
sys.path.append('..')

from dataclasses import dataclass
from sklearn.decomposition import PCA
from experiments.experimental_transformer import DemoTransformer

from server.database import SessionLocal
from server.model import Model
from server.prompt import Prompt
from server.resid import Resid
from server.utils import cuda, enc
from server.resid_writer import write_resids_for_prompt
from server.transformer import reference_gpt2

from transformer_lens import loading_from_pretrained as loading

import numpy as np
import matplotlib.pyplot as plt

def main():
    sess = SessionLocal()

    prompt_text = ' a' * 1000
    encoded_prompt_text = enc.encode(prompt_text)
    

    if not (prompt := sess.query(Prompt).filter(Prompt.text == prompt_text).one_or_none()):
        prompt = Prompt(
            text=prompt_text,
            encoded_text_split_by_token=encoded_prompt_text,
            length_in_tokens=len(encoded_prompt_text),
            dataset='aaa',
            text_split_by_token=[enc.decode([token]) for token in encoded_prompt_text],
        )

        sess.add(prompt)
        sess.commit()

    gpt2_small = sess.query(Model).filter(Model.name == "gpt2-small").one_or_none()

    write_resids_for_prompt(sess, prompt, gpt2_small, more_commits=True, verbose=True,
                            # keys=[f'blocks.{i}.hook_resid_pre' for i in range(12)])
                            keys=[*[f'blocks.{i}.ln1.hook_normalized' for i in range(12)], *[f'blocks.{i}.hook_resid_pre' for i in range(12)]])


def pca_resids():

    sess = SessionLocal()

    gpt2_small = sess.query(Model).filter(Model.name == "gpt2-small").one_or_none()

    for i in range(12):
        # type = f'blocks.{i}.hook_resid_pre'
        type = 'blocks.{i}.ln1.hook_normalized'

        resids = (
            sess.query(Resid)
            .filter(Resid.model == gpt2_small)
            .filter(Resid.dataset == 'aaa')
            .filter(Resid.type == type)
            .filter(Resid.layer == i)
            .filter(Resid.head == None)
            .filter(Resid.token_position > 0)
            .order_by(Resid.token_position.asc())
            .all()
        )

        matrix = np.array([resid.resid for resid in resids])

        # Apply PCA
        pca = PCA(n_components=3)
        matrix_pca = pca.fit_transform(matrix)

        # Create a scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Get a color map
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0, 1, len(matrix)))

        # Scatter plot
        sc = ax.scatter(matrix_pca[:, 0], matrix_pca[:, 1], matrix_pca[:, 2], c=colors)

        # Set labels
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        
        plt.colorbar(sc)

        plt.show()


def joint_pca_resids():

    sess = SessionLocal()

    gpt2_small = sess.query(Model).filter(Model.name == "gpt2-small").one_or_none()

    i = 2
    j = 3

    type1 = f'blocks.{i}.hook_resid_pre'
    type2 = f'blocks.{j}.hook_resid_pre'

    resids1 = (
        sess.query(Resid)
        .filter(Resid.model == gpt2_small)
        .filter(Resid.dataset == 'aaa')
        .filter(Resid.type == type1)
        .filter(Resid.layer == i)
        .filter(Resid.head == None)
        .filter(Resid.token_position > 0)
        .order_by(Resid.token_position.asc())
        .all()
    )


    # resids2 = (
    #     sess.query(Resid)
    #     .filter(Resid.model == gpt2_small)
    #     .filter(Resid.dataset == 'aaa')
    #     .filter(Resid.type == type2)
    #     .filter(Resid.layer == j)
    #     .filter(Resid.head == None)
    #     .filter(Resid.token_position > 0)
    #     .order_by(Resid.token_position.asc())
    #     .all()
    # )

    # prompt = sess.query(Prompt).get(776067)    
    # resids2 = (
    #     sess.query(Resid)
    #     .filter(Resid.model == gpt2_small)
    #     .filter(Resid.prompt == prompt)
    #     .filter(Resid.type == type1)
    #     .filter(Resid.layer == i)
    #     .filter(Resid.head == None)
    #     .filter(Resid.token_position > 0)
    #     .order_by(Resid.token_position.asc())
    #     .all()
    # )

    resids2 = (
        sess.query(Resid)
        .filter(Resid.model == gpt2_small)
        .filter(Resid.dataset == 'aaa_average_in_blocks_10')
        .filter(Resid.type == type1)
        .filter(Resid.layer == i)
        .filter(Resid.head == None)
        .filter(Resid.token_position > 0)
        .order_by(Resid.token_position.asc())
        .all()
    )

    matrix1 = np.array([resid.resid for resid in resids1])
    matrix2 = np.array([resid.resid for resid in resids2])

    # Fit the PCA on matrix1
    pca = PCA(n_components=3)
    pca.fit(matrix1)

    # Apply the PCA transformation to matrix2
    # matrix2_pca = pca.transform(matrix2)

    # # Create a scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # # Get a color map
    # cmap = plt.get_cmap("viridis")
    # colors = cmap(np.linspace(0, 1, len(matrix2)))

    # # Scatter plot
    # sc = ax.scatter(matrix2_pca[:, 0], matrix2_pca[:, 1], matrix2_pca[:, 2], c=colors)

    # # Set labels
    # ax.set_xlabel('PCA 1')
    # ax.set_ylabel('PCA 2')
    # ax.set_zlabel('PCA 3')

    # plt.colorbar(sc)

    # plt.show()

    # Apply the PCA transformation to matrix2
    matrix1_pca = pca.transform(matrix1)

    # # Create a scatter plot
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # Get a color map
    cmap = plt.get_cmap("jet")
    colors = cmap(np.linspace(0, 1, len(matrix1)))

    # Scatter plot
    sc = ax.scatter(matrix1_pca[:, 0], matrix1_pca[:, 1], matrix1_pca[:, 2], c=colors)

    # Set labels
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')

    # plt.colorbar(sc)

    plt.show()





def attn_head_eigs():
    sess = SessionLocal()

    gpt2_small = sess.query(Model).filter(Model.name == "gpt2-small").one_or_none()

    for i in range(12):
        # type = f'blocks.{i}.hook_resid_pre'
        type = 'blocks.{i}.ln1.hook_normalized'

        resids = (
            sess.query(Resid)
            .filter(Resid.model == gpt2_small)
            .filter(Resid.dataset == 'abab')
            .filter(Resid.type == type)
            .filter(Resid.layer == i)
            .filter(Resid.head == None)
            .filter(Resid.token_position > 0)
            .order_by(Resid.token_position.asc())
            .all()
        )

        layer = 4
        head_num = 11

        W_Q = reference_gpt2.blocks[layer].attn.W_Q.detach().numpy()[head_num]
        W_K = reference_gpt2.blocks[layer].attn.W_K.detach().numpy()[head_num]

        resids_mat = np.array([resid.resid for resid in resids])

        matrix = np.dot(W_K, np.transpose(W_Q))

        print(matrix.shape)
        print(W_Q.shape)
        print(W_K.shape)

        # Compute eigenvectors and eigenvalues of the matrix
        eigvals, eigvecs = np.linalg.eig(matrix)
        # Get indices of the largest three eigenvalues
        idx = np.abs(eigvals).argsort()[-3:][::-1]
        # Select the eigenvectors associated with the largest three eigenvalues
        eigvecs = eigvecs[:, idx]

        # Project each row of resids_mat onto the three eigenvectors
        resids_proj = np.dot(resids_mat, eigvecs)

        # Create a scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Get a color map
        cmap = plt.get_cmap("jet")
        colors = cmap(np.linspace(0, 1, len(resids_mat)))

        # Scatter plot
        sc = ax.scatter(resids_proj[:, 0], resids_proj[:, 1], resids_proj[:, 2], c=colors)

        # Set labels
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        plt.colorbar(sc)

        plt.show()


def experimental_add_resids_for_aaaa():
    sess = SessionLocal()

    prompt1 = sess.query(Prompt).filter(Prompt.dataset == 'aaa').first()

    reference_text = prompt1.text

    print(prompt1.text)

    # reference_text = ' a b b' * 200

    # reference_text = '\n\n 0 0' * 200
    # reference_text = '\n\n a a' * 200
    # reference_text = ' 1 0 0' * 200

    print([enc.decode([tok]) for tok in enc.encode(reference_text)][150:200])

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

    model_name = 'gpt2-small'
    demo_gpt2 = DemoTransformer(get_basic_config(model_name=model_name))
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

    tokens = cuda(tokens)

    logits = demo_gpt2(tokens, 
                       average_pos_embed_in_blocks=10, 
                       write_resid_keys_for_prompt=(sess, 'aaa_average_in_blocks_10', prompt1, [*[f'blocks.{i}.ln1.hook_normalized' for i in range(12)], *[f'blocks.{i}.hook_resid_pre' for i in range(12)]],),
                    # zero_out_pos=500,
                    # zero_out_specific_head=heads_to_zero_out,
                    # zero_out_specific_head=[*[(2, h) for h in range(12)], *[(3, h) for h in range(12)]],
                    # zero_out_specific_head=[*[(2, h) for h in range(12)], *[(1, h) for h in range(12)]],
                    # zero_out_specific_head=(None,4),
                    # save_attn_patterns_filename='long_repeat_prompt_no_embed',
                    # no_pos_embed_contribution = True,
                    # no_embed_contribution = True,
                    # permute_pos_embed=True,
                    )


if __name__ == '__main__':
    # main()
    # experimental_add_resids_for_aaaa()
    # pca_resids()
    joint_pca_resids()
    # attn_head_eigs()