# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Step 11: Load Weights and Run Model

Load pretrained GPT-2 weights from HuggingFace and run the complete model.

Tasks:
1. Load HuggingFace GPT-2 model and weights
2. Initialize MAX model and load state dict
3. Transpose weights for Conv1D->Linear compatibility
4. Compile model with correct input specification
5. Create interactive generation loop

Run: pixi run s11
"""
from typing import cast
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.tensor import TensorType, defaults, default_device, default_dtype
from max.nn import Linear, Module
from step_01 import GPT2Config
from step_08 import MaxGPT2LMHeadModel
from step_10 import generate_text

def run_model() -> None:
    """Load GPT-2 model, compile it, and run interactive text generation."""

    _, device = defaults()
    dtype = DType.float32  # Apple GPU does not support bfloat16
    # device = CPU()
    # dtype = DType.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # TODO: Load HuggingFace model
    # Hint: hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Hint: print(f"Loaded HuggingFace model:\n{hf_model}")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    print(f"Loaded HuggingFace model:\n{hf_model}")

    # TODO: Initialize MAX model with device
    # Hint: _, device = defaults()
    # Hint: print(f"Using device: {device}")
    # Hint: config = GPT2Config()
    # Hint: max_model = MaxGPT2LMHeadModel(config)

    # Load HuggingFace model weights on CPU device
    with default_device(CPU()), default_dtype(dtype):
        config = GPT2Config()
        max_model = MaxGPT2LMHeadModel(config)

        print(
            f"Model has {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding dim"
        )

        # TODO: Transpose weights for Linear layers before loading to Max
        # Hint: HuggingFace uses Conv1D which stores weights transposed
        weights = {}
        for name, param in hf_model.state_dict().items():
            if any(layer_name in name for layer_name in ["c_attn", "c_proj", "c_fc"]) and not name.endswith(".bias"):
                print(f"Transposing {name}: {param.shape}")
                weights[name] = param.T.contiguous()
            else:
                weights[name] = param

        max_model.load_state_dict(weights)

    # Move model to GPU device
    max_model.to(device)

    # TODO: Load state dict and move to device
    # Hint: max_model.load_state_dict(weights)
    # Hint: max_model.to(device)
    # TODO: Transpose weights for Linear layers
    # Hint: HuggingFace uses Conv1D which stores weights transposed
    # Hint: for name, child in max_model.descendents:
    #     if isinstance(child, Linear):
    #         if any(layer_name in name for layer_name in ["c_attn", "c_proj", "c_fc"]):
    #             print(f"Transposing {name}: {child.weight.shape}")
    #             child.weight = child.weight.T
    # for name, child in max_model.descendants:
    #     if isinstance(child, Linear):
    #         if any(layer_name in name for layer_name in ["c_attn", "c_proj", "c_fc"]):
    #             print(f"Transposing {name}: {child.weight.shape}")
    #             child.weight = child.weight.T

    # TODO: Initialize tokenizer
    # Hint: tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Hint: tokenizer.pad_token = tokenizer.eos_token
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # TODO: Compile model
    # Hint: print("\nCompiling model...")
    # Hint: Create TensorType with shape ("batch", "seqlen") and int64 dtype
    # Hint: token_type = TensorType(DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device))
    # Hint: compiled_max_model = max_model.compile(token_type)
    print("\nCompiling model...")
    token_type = TensorType(DType.int64, ("batch", "seqlen"), device=DeviceRef.from_device(device))
    compiled_max_model = max_model.compile(token_type)

    # Interactive prompt loop
    print("\n" + "=" * 50)
    print("Model ready! Enter prompts to generate text.")
    print("Press Ctrl+C or type 'quit' to exit.")
    print("=" * 50 + "\n")

    # TODO: Implement interactive generation loop
    # Hint: try:
    #     while True:
    #         user_input = input("Enter your prompt: ").strip()
    #         if user_input.lower() in ['quit', 'exit', 'q']:
    #             break
    #         if not user_input:
    #             continue
    #         generated_text = generate_text(
    #             compiled_max_model, tokenizer, device, user_input,
    #             max_new_tokens=50, temperature=0.8, do_sample=True
    #         )
    #         print(f"\nGenerated text:\n{generated_text}\n")
    # except KeyboardInterrupt:
    #     print("\n\nExiting...")
    try:
        while True:
            user_input = input("Enter your prompt: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
            generated_text = generate_text(
                cast(Module, compiled_max_model), tokenizer, device, user_input,
                max_new_tokens=50, temperature=0.8, do_sample=True
            )
            print(f"\nGenerated text:\n{generated_text}\n")
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    run_model()
