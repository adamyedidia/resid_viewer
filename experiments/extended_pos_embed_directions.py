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

if __name__ == '__main__':
    main()