"""Inspect how the Qwen tokenizer represents control-engineering terms."""

from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

EXAMPLES = [
    "the",
    "of",
    "atakan",
    "nonlinear dynamic inversion",
]


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Tokenizer class: {type(tokenizer).__name__}")
print(f"Vocabulary size: {len(tokenizer):,}")

for text in EXAMPLES:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    readable_pieces = [tokenizer.decode([token_id]) for token_id in token_ids]

    print(f"\nText: {text!r}")
    print(f"Token count: {len(token_ids)}")
    print(f"Token IDs: {token_ids}")
    print(f"Raw tokens: {raw_tokens}")
    print(f"Decoded pieces: {readable_pieces}")
