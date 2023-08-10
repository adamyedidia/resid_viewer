import sys
sys.path.append('..')

from server.database import SessionLocal
from server.extended_pos_embed import ExtendedPosEmbed, get_extended_pos_embed_matrix
from server.model import Model
from server.transformer import reference_gpt2
from server.utils import enc, is_alphanumeric_spaces_or_punctuation

import matplotlib.pyplot as plt
import numpy as np


def main():
    sess = SessionLocal()

    model = sess.query(Model).filter(Model.name == 'gpt2-small').first()

    if model is None:
        raise Exception('No model found')

    TOKEN_POSITION = 500

    # plt.matshow(get_extended_pos_embed_matrix(sess, model, 2, 'blocks.2.ln1.hook_normalized'))
    # plt.colorbar()
    # plt.show()

    matrix = get_extended_pos_embed_matrix(sess, model, 2, 'blocks.2.ln1.hook_normalized')
    print(matrix.shape)
    s = set()
    for j in range(768):
        if max(
            [abs(row[j]) for row in matrix]
        ) > 0.3:
            s.add(j)

    print(s)

    plt.matshow(get_extended_pos_embed_matrix(sess, model, 2, 'blocks.2.ln1.hook_normalized'))
    plt.colorbar()
    plt.show()

    plt.matshow(reference_gpt2.W_pos.detach().numpy())
    plt.colorbar()
    plt.show()    

    matrix = reference_gpt2.W_pos.detach().numpy()

    s = set()
    for j in range(768):
        if max(
            [abs(row[j]) for row in matrix]
        ) > 0.3:
            s.add(j)

    print(s)


    pos_embed1 = (
        sess.query(ExtendedPosEmbed)
        .filter(ExtendedPosEmbed.model == model)
        .filter(ExtendedPosEmbed.layer == 2)
        .filter(ExtendedPosEmbed.type == 'blocks.2.ln1.hook_normalized')
        .filter(ExtendedPosEmbed.token_position == TOKEN_POSITION)
        .order_by(ExtendedPosEmbed.created_at.desc())
        .first()
    )

    pos_embed2 = (
        sess.query(ExtendedPosEmbed)
        .filter(ExtendedPosEmbed.model == model)
        .filter(ExtendedPosEmbed.layer == 2)
        .filter(ExtendedPosEmbed.type == 'blocks.2.ln1.hook_normalized')
        .filter(ExtendedPosEmbed.token_position == TOKEN_POSITION - 1)
        .order_by(ExtendedPosEmbed.created_at.desc())
        .first()        
    )

    if pos_embed1 is None or pos_embed2 is None:
        raise Exception('No pos embeds found')

    direction = np.array(pos_embed1.extended_pos_embed) - np.array(pos_embed2.extended_pos_embed)

    biggest_pos_dot_product = -10000
    biggest_neg_dot_product = 10000
    biggest_pos_dot_product_index = None
    biggest_neg_dot_product_index = None

    for char_index in range(50256):
        decoded_token = enc.decode([char_index])
        if not is_alphanumeric_spaces_or_punctuation(decoded_token):
            continue

        embed = reference_gpt2.W_E.detach().numpy()[char_index]

        dot_product = direction.dot(embed)

        if dot_product > biggest_pos_dot_product:
            biggest_pos_dot_product = dot_product
            biggest_pos_dot_product_index = char_index

        if dot_product < biggest_neg_dot_product:
            biggest_neg_dot_product = dot_product
            biggest_neg_dot_product_index = char_index
    
    assert biggest_pos_dot_product_index
    assert biggest_neg_dot_product_index

    print(biggest_pos_dot_product_index, biggest_pos_dot_product, enc.decode([biggest_pos_dot_product_index]))
    print(biggest_neg_dot_product_index, biggest_neg_dot_product, enc.decode([biggest_neg_dot_product_index]))

def get_canonical_angles_between_extended_pos_embed_and_pos_embed():
    sess = SessionLocal()

    model = sess.query(Model).filter(Model.name == 'gpt2-small').first()

    if model is None:
        return

    positional_embedding_matrix = reference_gpt2.W_pos.detach().numpy()

    extended_positional_embedding_matrix = get_extended_pos_embed_matrix(sess, model, 2, 'blocks.2.ln1.hook_normalized')

    A = positional_embedding_matrix
    B = extended_positional_embedding_matrix

    print(A.shape)
    print(B.shape)

    # Assume A and B are your matrices
    U_A, s_A, Vh_A = np.linalg.svd(A, full_matrices=False)
    U_B, s_B, Vh_B = np.linalg.svd(B, full_matrices=False)

    # Let's consider the first k largest singular vectors
    k = 3  # or any other number you choose
    U_A_k = U_A[:, :k]
    U_B_k = U_B[:, :k]

    # Compute the matrix of cosines of canonical angles
    C = U_A_k.T @ U_B_k

    # Singular values of C give the cosine values of canonical angles
    sigma = np.linalg.svd(C, compute_uv=False)

    print(sigma)

    # Get the actual canonical angles
    theta = np.arccos(np.clip(sigma, -1, 1))

    def cosine_similarity(vec1, vec2):
        # Compute the dot product between vec1 and vec2
        dot_product = np.dot(vec1, vec2)
        
        # Compute the L2 norm (Euclidean norm) of vec1 and vec2
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        # Compute the cosine similarity
        similarity = dot_product / (norm_vec1 * norm_vec2)
    
        return similarity
    
    print(cosine_similarity(U_A_k[:, 0], U_B_k[:, 0]))

    # plt.plot(U_A_k[:, 0])
    # plt.plot(U_B_k[:, 0])
    # plt.show()

    print(theta)


if __name__ == '__main__':
    # main()
    get_canonical_angles_between_extended_pos_embed_and_pos_embed()

