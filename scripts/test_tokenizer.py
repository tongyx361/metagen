#!/usr/bin/env python3
"""
Test script to verify if the DeepSeek tokenizer adds a BOS token by default.
"""

from transformers import AutoTokenizer


def test_bos_token():
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )

    # Test string
    test_text = "Hello, how are you?"

    # Tokenize without special tokens
    tokens_without_special = tokenizer(test_text, add_special_tokens=False)
    detokenized_without_special = tokenizer.decode(tokens_without_special["input_ids"])

    # Tokenize with special tokens (default behavior)
    tokens_with_special = tokenizer(test_text)
    detokenized_with_special = tokenizer.decode(tokens_with_special["input_ids"])

    # Print results
    print(f"Test text: {test_text}")
    print(f"Tokens without special tokens: {tokens_without_special['input_ids']}")
    print(f"Detokenized without special tokens: {detokenized_without_special}")
    print(f"Tokens with special tokens: {tokens_with_special['input_ids']}")
    print(f"Detokenized with special tokens: {detokenized_with_special}")

    # Check if BOS token is added
    if len(tokens_with_special["input_ids"]) > len(tokens_without_special["input_ids"]):
        print("\nThe tokenizer DOES add special tokens by default")
        print(f"First token in special encoding: {tokens_with_special['input_ids'][0]}")
        print(f"BOS token ID: {tokenizer.bos_token_id}")
    else:
        print("\nThe tokenizer does NOT add special tokens by default")


if __name__ == "__main__":
    test_bos_token()
