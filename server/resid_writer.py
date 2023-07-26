import sys
sys.path.append('..')

from sqlalchemy import and_, exists
from server.database import SessionLocal
from server.model import Model
from server.prompt import Prompt
from server.resid import Resid, add_resid
from server.utils import cuda, enc, get_layer_num_from_resid_type
from server.transformer import reference_gpt2, model_name

from typing import Optional


def write_resids_for_prompt(sess, prompt: Prompt, model: Model, more_commits = False,
                            verbose = False, keys: Optional[list] = None) -> None:
    text = prompt.text
    tokens = reference_gpt2.to_tokens(text)  # type: ignore
    tokens = cuda(tokens)
    _, cache = reference_gpt2.run_with_cache(tokens)

    for key in keys if keys is not None else cache.keys():
        if verbose:
            print(key)
        value = cache[key]
        shape = value.shape
        assert shape[0] == 1
        
        layer_num = get_layer_num_from_resid_type(key)

        if shape[1] == 12:
            assert len(shape) == 4
            assert shape[2] == prompt.length_in_tokens + 1

            for i in range(12):
                for j in range(prompt.length_in_tokens + 1):  # type: ignore
                    resid = value[0, i, j, :].detach().numpy()

                    add_resid(
                        sess,
                        resid,
                        model,
                        prompt,
                        layer_num,
                        key,
                        j,
                        i,
                        no_commit=True,
                        skip_dedupe_check=True,
                    )

                if more_commits:
                    sess.commit()

        if shape[1] == prompt.length_in_tokens + 1:
            if shape[2] == 12:
                assert len(shape) == 4, shape
                for i in range(12):
                    for j in range(prompt.length_in_tokens + 1):  # type: ignore
                        resid = value[0, j, i, :].detach().numpy()

                        add_resid(
                            sess,
                            resid,
                            model,
                            prompt,
                            layer_num,
                            key,
                            j,
                            i,
                            no_commit=True,
                            skip_dedupe_check=True,
                        )          

                    if more_commits:
                        sess.commit()          

            else:
                assert len(shape) == 3, shape

                for i in range(prompt.length_in_tokens + 1):  # type: ignore
                    resid = value[0, i, :].detach().numpy()

                    add_resid(
                        sess,
                        resid,
                        model,
                        prompt,
                        layer_num,
                        key,
                        i,
                        None,
                        no_commit=True,
                        skip_dedupe_check=True,
                    )

                if more_commits:
                    sess.commit()

        sess.commit()


def main():

    sess = SessionLocal()

    prompts_to_populate = (
        sess.query(Prompt)
        .filter(Prompt.length_in_tokens == 30)
        # .filter(Prompt.length_in_tokens == 1023)
        .all()
    )

    model = sess.query(Model).filter(Model.name == model_name).one_or_none()

    if model is None:
        model = Model(name=model_name)
        sess.add(model)
        sess.commit()

    if not model:
        return

    for i, prompt in enumerate(prompts_to_populate):
        print(f'Writing resids for {prompt}, {i}')
        
        if sess.query(exists().where(and_(Resid.prompt_id == prompt.id,
                                          Resid.model_id == model.id))).scalar():
            print(f'Already wrote resids for {prompt}, {i}')
            continue

        try:
            write_resids_for_prompt(sess, prompt, model, more_commits=True, verbose=True)

        except AssertionError:
            print(f'Failed to write resids for {prompt}, {i}')
            sess.rollback()
            continue


if __name__ == '__main__':
    main()

