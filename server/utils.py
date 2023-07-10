from typing import Optional
import tiktoken

from settings import M1_MAC

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

