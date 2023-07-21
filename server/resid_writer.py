import sys
sys.path.append('..')

from sqlalchemy import and_, exists
from server.database import SessionLocal
from server.model import Model
from server.prompt import Prompt
from server.resid import Resid, add_resid
from server.utils import cuda, enc, get_layer_num_from_resid_type
from server.transformer import reference_gpt2, model_name

def write_resids_for_prompt(sess, prompt: Prompt, model: Model) -> None:
    text = prompt.text
    tokens = reference_gpt2.to_tokens(text)  # type: ignore
    tokens = cuda(tokens)
    _, cache = reference_gpt2.run_with_cache(tokens)

    for key in cache.keys():
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

        sess.commit()


def main():

    sess = SessionLocal()

    prompts_to_populate = (
        sess.query(Prompt)
        .filter(Prompt.length_in_tokens == 30)
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
            write_resids_for_prompt(sess, prompt, model)

        except AssertionError:
            print(f'Failed to write resids for {prompt}, {i}')
            sess.rollback()
            continue


    # reference_text = "The greatest president of all time was Abraham"
    # tokens = reference_gpt2.to_tokens(reference_text)

    # def cuda(x):
    #     return x.to('cpu') if M1_MAC else x.cuda()

    # tokens = cuda(tokens)
    # logits, cache = reference_gpt2.run_with_cache(tokens)

    # print(cache.keys())
    # print(cache['blocks.3.attn.hook_v'].shape)
    # print(len(cache.keys()))

    # last_logits = logits[-1, -1]  # type: ignore
    # # Apply softmax to convert the logits to probabilities
    # probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()
    
    # # Get the indices of the top 10 probabilities
    # topk_indices = np.argpartition(probabilities, -10)[-10:]
    # # Get the top 10 probabilities
    # topk_probabilities = probabilities[topk_indices]
    # # Get the top 10 tokens
    # topk_tokens = [enc.decode([i]) for i in topk_indices]

    # # Print the top 10 tokens and their probabilities
    # for token, probability in zip(topk_tokens, topk_probabilities):
    #     print(f"Token: {token}, Probability: {probability}")


if __name__ == '__main__':
    # reference_text = "The greatest president of all time was Abraham"
    # reference_text = 'hey, what is that? is that a dog'
    # tokens = reference_gpt2.to_tokens(reference_text)

    # def cuda(x):
    #     return x.to('cpu') if M1_MAC else x.cuda()

    # tokens = cuda(tokens)
    # logits, cache = reference_gpt2.run_with_cache(tokens)

    # print(cache.keys())
    # print(cache['blocks.3.attn.hook_v'].shape)
    # print(len(cache.keys()))

    # last_logits = logits[-1, -1]  # type: ignore
    # # Apply softmax to convert the logits to probabilities
    # probabilities = torch.nn.functional.softmax(last_logits, dim=0).detach().numpy()
    
    # # Get the indices of the top 10 probabilities
    # topk_indices = np.argpartition(probabilities, -10)[-10:]
    # # Get the top 10 probabilities
    # topk_probabilities = probabilities[topk_indices]
    # # Get the top 10 tokens
    # topk_tokens = [enc.decode([i]) for i in topk_indices]

    # # Print the top 10 tokens and their probabilities
    # for token, probability in zip(topk_tokens, topk_probabilities):
    #     print(f"Token: {token}, Probability: {probability}")

    # raise Exception()

    main()

