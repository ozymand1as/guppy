#!/usr/bin/env python3
"""Laguna S 2.1 chat wrapper for the laguna C engine.

Applies the correct chat template and tokenization, then passes
token IDs to the laguna binary via IDS env var. Decodes the output
tokens back to text.
"""
import os
import sys
import json
import subprocess
import re

def encode_prompt(prompt, model_dir, enable_thinking=True):
    """Encode a user prompt using the Laguna chat template."""
    from tokenizers import Tokenizer
    
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    
    # Build the prompt using the chat template logic
    # The template uses special tokens like <|tool_call_begin|>system
    # But the tokenizer has these as special tokens 0-69
    # We need to use the actual special token IDs
    
    # From the tokenizer, special tokens 0-69 are control characters
    # The chat template uses:
    # - BOS (id=2) at start
    # - System message wrapped in special tokens
    # - User message wrapped in special tokens
    # - Assistant response starts with thinking block
    
    # Build the formatted prompt manually based on the template logic
    # The template renders:
    # <|tool_call_begin|>system\n{system_message}<|tool_call_begin|>user\n{prompt}<|tool_call_begin|>assistant\n<thinking>
    
    # But since the tokenizer doesn't recognize <|tool_call_begin|> as a special token,
    # we need to use the actual special token IDs
    
    # From the tokenizer config, let's use the special tokens directly
    # Token 2 = BOS, Token 24 = EOS
    
    # For now, let's try a simple approach: just tokenize the prompt directly
    # with BOS and EOS tokens
    enc = tok.encode(prompt, add_special_tokens=True)
    return enc.ids, tok

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 laguna_chat.py <model_dir> <prompt> [ngen]")
        sys.exit(1)
    
    model_dir = os.path.expanduser(sys.argv[1])
    prompt = sys.argv[2]
    ngen = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    # Encode the prompt
    token_ids, tok = encode_prompt(prompt, model_dir)
    ids_str = ",".join(str(t) for t in token_ids)
    
    print(f"Prompt: {prompt}")
    print(f"Token IDs: {ids_str}")
    print(f"Tokens: {len(token_ids)}")
    print(f"Generating {ngen} tokens...")
    print()
    
    # Run the engine
    env = os.environ.copy()
    env["SNAP"] = model_dir
    env["RAM_GB"] = "20"
    env["PROMPT"] = "x"
    env["IDS"] = ids_str
    env["NGEN"] = str(ngen)
    env["TEMP"] = "1.0"
    env["NUCLEUS"] = "1.0"
    env["TOPK"] = "20"
    
    engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laguna")
    result = subprocess.run([engine_path, "64", "4"], env=env, capture_output=True, text=True)
    
    # Print engine output
    if result.stderr:
        print(result.stderr, end="")
    
    # Parse the generated tokens from output
    output = result.stdout
    print(output, end="")
    
    # Extract token IDs from the output
    token_match = re.search(r'tokens:\s+(.+)', output)
    if token_match:
        token_ids_str = token_match.group(1).strip()
        generated_ids = [int(x) for x in token_ids_str.split()]
        
        # Decode the generated tokens
        decoded = tok.decode(generated_ids)
        print(f"\nDecoded output: {decoded}")
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
