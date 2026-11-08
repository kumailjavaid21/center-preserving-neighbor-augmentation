import hashlib
SEED=20261029

def named_seed(stream, *keys):
    value = '|'.join(map(str, (SEED, stream, *keys))).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') % (2**32)
