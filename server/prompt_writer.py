import sys
sys.path.append('..')

from server.prompt import Prompt
from server.utils import enc
from server.database import SessionLocal
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
    for i in range(int(sys.argv[1]), int(sys.argv[2])):  # type: ignore
        print(f'Writing prompt {i} of {len(ds)} of length {prompt_length}')  # type: ignore

        item = ds[i]  # type: ignore
        text: str = item['text']  # type: ignore
        encoded_text = enc.encode(text)
        # subprompts = split_prompt_by_tokens(encoded_text, prompt_length)

        truncated_encoded_text = encoded_text[:prompt_length]
        subprompt_singletext = enc.decode(truncated_encoded_text)
        subprompt_decoded_by_token = [enc.decode([token]) for token in truncated_encoded_text]

        if (existing_prompt := sess.query(Prompt).filter(Prompt.text == subprompt_singletext).first()) is not None:
            # print(f'Prompt already exists: {existing_prompt}')
            continue

        sess.add(Prompt(
            text=subprompt_singletext,
            encoded_text_split_by_token=truncated_encoded_text,
            text_split_by_token=subprompt_decoded_by_token,
            length_in_tokens=len(truncated_encoded_text),
            dataset=dataset_short_name,
        ))
        sess.commit()

        
def write_catdog_prompts() -> None:
    sess = SessionLocal()
    print('Writing catdog prompts')

    seq_length = 20

    for i in range(seq_length):
        dog_prompt_long = ' dog' * i + (' cat' * seq_length + ' dog' * seq_length) * 30
        cat_prompt_long = ' cat' * i + (' dog' * seq_length + ' cat' * seq_length) * 30

        dog_prompt_encoded_long = enc.encode(dog_prompt_long)
        cat_prompt_encoded_long = enc.encode(cat_prompt_long)

        dog_prompt_encoded = dog_prompt_encoded_long[:1023]
        cat_prompt_encoded = cat_prompt_encoded_long[:1023]

        dog_prompt_str_split_by_token = [enc.decode([token]) for token in dog_prompt_encoded]
        cat_prompt_str_split_by_token = [enc.decode([token]) for token in cat_prompt_encoded]

        dog_prompt_text = enc.decode(dog_prompt_encoded)
        cat_prompt_text = enc.decode(cat_prompt_encoded)

        if (existing_prompt := sess.query(Prompt).filter(Prompt.text == dog_prompt_text).first()) is not None:
            # print(f'Prompt already exists: {existing_prompt}')
            continue

        sess.add(Prompt(
            text=dog_prompt_text,
            encoded_text_split_by_token=dog_prompt_encoded,
            text_split_by_token=dog_prompt_str_split_by_token,
            length_in_tokens=len(dog_prompt_encoded),
            dataset='catdog',
        ))

        sess.add(Prompt(
            text=cat_prompt_text,
            encoded_text_split_by_token=cat_prompt_encoded,
            text_split_by_token=cat_prompt_str_split_by_token,
            length_in_tokens=len(cat_prompt_encoded),
            dataset='catdog',
        ))

        sess.commit()


if __name__ == '__main__':
    write_openwebtext10k_prompts()
    # write_catdog_prompts()