from prompt import Prompt
from utils import enc
from database import SessionLocal
import random

def split_prompt_by_tokens(encoded_text: list[int], num_tokens: int) -> list[list[int]]:
    # splits the encoded_text into lists each of length num_tokens

    list_of_subprompts = [encoded_text[i:i+num_tokens] for i in range(0, len(encoded_text), num_tokens)]
    return [l for l in list_of_subprompts if len(l) == num_tokens]


def write_openwebtext10k_prompts() -> None:
    from datasets import load_dataset
    dataset_name = "stas/openwebtext-10k"
    dataset_short_name = dataset_name.split('/')[-1]
    ds = load_dataset(dataset_name, split='train')

    sess = SessionLocal()

    counter = 0

    prompt_length = 30
    for i in range(len(ds)):  # type: ignore
        print(f'Writing prompt {i} of {len(ds)} of length {prompt_length}')  # type: ignore

        item = ds[i]  # type: ignore
        text: str = item['text']  # type: ignore
        encoded_text = enc.encode(text)
        subprompts = split_prompt_by_tokens(encoded_text, prompt_length)
        for subprompt in subprompts:
            # deterministically subsample 10% of the prompts
            counter += 1
            if counter % 10 != 0:
                continue

            subprompt_singletext = enc.decode(subprompt)
            subprompt_decoded_by_token = [enc.decode([token]) for token in subprompt]

            if (existing_prompt := sess.query(Prompt).filter(Prompt.text == subprompt_singletext).first()) is not None:
                # print(f'Prompt already exists: {existing_prompt}')
                continue

            sess.add(Prompt(
                text=subprompt_singletext,
                encoded_text_split_by_token=subprompt,
                text_split_by_token=subprompt_decoded_by_token,
                length_in_tokens=len(subprompt),
                dataset=dataset_short_name,
            ))
            sess.commit()


if __name__ == '__main__':
    write_openwebtext10k_prompts()