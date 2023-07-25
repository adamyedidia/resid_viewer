from typing import Optional
import tiktoken
import random

from server.settings import M1_MAC

enc = tiktoken.get_encoding('r50k_base')

def cuda(x):
    return x.to('cpu') if M1_MAC else x.cuda()


def is_numeric_string(s) -> bool:
    return_value = True
    for c in s:
        if c not in '0123456789':
            return_value = False
            break

    return return_value


def get_layer_num_from_resid_type(resid_type: str) -> Optional[int]:
    split_type = resid_type.split('.')
    for s in split_type:
        if is_numeric_string(s):
            return int(s)
    return None


def get_random_token(exclude_endoftext: bool = True) -> int:
    return random.randint(0, 50255 if exclude_endoftext else 50256)


def get_random_str_token(exclude_endoftext: bool = True) -> str:
    return enc.decode([get_random_token(exclude_endoftext)])


def lists_are_equal(l1, l2):
    return all([x == y for x, y in zip(l1, l2)])


def has_unique_tokenization(l: list[int]) -> bool:
    return lists_are_equal(l, enc.encode(enc.decode(l)))
