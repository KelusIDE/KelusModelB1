"""Use the repository's Rust BPE encoder while preserving checkpoint/tokenizer formats."""


def accelerated(tokenizer):
    import kani_tokenize
    value = tokenizer.to_dict()
    if value.get('kind') != 'bpe' or value.get('version') not in (2, 3):
        raise ValueError('Rust acceleration needs Kani BPE version 2 or 3')
    rust = kani_tokenize.Tokenizer([tuple(pair) for pair in value['merges']], value['version'])

    class Encoder:
        encode = rust.encode

        def __getattr__(self, name):
            return getattr(tokenizer, name)

    return Encoder()
