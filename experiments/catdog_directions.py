import sys
sys.path.append('..')

from server.database import SessionLocal
from server.prompt import Prompt
from server.direction import Direction
from server.resid import Resid
from server.model import Model
import numpy as np
import matplotlib.pyplot as plt


def main():
    sess = SessionLocal()

    prompt = (
        sess.query(Prompt)
        .filter(Prompt.dataset == 'catdog')
        .filter(Prompt.length_in_tokens == 1023)
        .first()
    )

    model = (
        sess.query(Model)
        .filter(Model.name == 'gpt2-small')
        .one_or_none()
    )

    if not prompt or not model:
        return

    print(prompt.text)

    directions = (
        sess.query(Direction)
        .filter(Direction.generated_by_process == 'catdog_pca')
        .filter(Direction.layer == 0)
        .filter(Direction.head == None)
        .filter(Direction.type == 'blocks.0.hook_attn_out')
        .order_by(Direction.component_index.asc())
        .all()
    )

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

    print(len(directions))
    print(len(resids))

    for i, direction in enumerate(directions[:10]):
        print(i)

        dot_prods = []
        for resid in resids:
            dot_prods.append(np.dot(np.array(resid.resid), np.array(direction.direction)))

        plt.plot(dot_prods)
        plt.show()
        

    sess.close()

if __name__ == "__main__":
    main()
    