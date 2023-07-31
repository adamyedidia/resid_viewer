import sys
sys.path.append('..')

from server.database import SessionLocal
from server.model import Model
from server.resid import Resid
from server.transformer import reference_gpt2

import numpy as np

def main():
    sess = SessionLocal()



    for layer_num in range(12):

        model = sess.query(Model).filter(Model.name == 'gpt2-small').one_or_none()

        resids = (
            sess.query(Resid)
            .filter(Resid.dataset == 'openwebtext-10k_long')
            .filter(Resid.type == f'blocks.{layer_num}.hook_resid_pre')
            .filter(Resid.layer == layer_num)
            .filter(Resid.model == model)
            .filter(Resid.token_position > 1) # The leading |<endoftext>| token is weird
            .all()
        )

        inputs = []
        outputs = []

        for resid in resids:
            inputs.append(resid.resid)  # type: ignore
            # print(resid.prompt.length_in_tokens)
            # print(resid.token_position)
            # print(resid)
            # print(resid.decoded_token)
            outputs.append(reference_gpt2.W_E.detach().numpy()[resid.prompt.encoded_text_split_by_token[resid.token_position - 2]])  # type: ignore

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        print(np.shape(inputs))
        print(np.shape(outputs))

        # Get a linear transformation from inputs to outputs

        learned_mat = np.dot(np.linalg.pinv(inputs), outputs)

        # Find the total error

        print(layer_num, np.linalg.norm(np.dot(inputs, learned_mat) - outputs))




if __name__ == '__main__':
    main()