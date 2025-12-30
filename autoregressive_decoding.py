import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_cache_length(past_key_values):
    """
    Get the cache length from past_key_values, handling different cache types.
    
    Args:
        past_key_values: Cache object (DynamicCache, tuple, list, or None)
    
    Returns:
        Cache length as int or string description, or 'N/A' if unable to determine
    """
    if past_key_values is None:
        return None
    
    if isinstance(past_key_values, (list, tuple)):
        # Tuple/list format: number of layers
        cache_length = f"{len(past_key_values)} layers"
        # Try to get sequence length from first layer if available
        if len(past_key_values) > 0 and isinstance(past_key_values[0], (list, tuple)) and len(past_key_values[0]) > 0:
            try:
                seq_len = past_key_values[0][0].shape[-2]
                cache_length += f", seq_len: {seq_len}"
            except:
                pass
        return cache_length
    elif hasattr(past_key_values, 'get_seq_length'):
        # DynamicCache: use get_seq_length() method
        return past_key_values.get_seq_length()
    elif hasattr(past_key_values, 'key_cache'):
        # DynamicCache: get seq length from key_cache
        try:
            if len(past_key_values.key_cache) > 0:
                return past_key_values.key_cache[0].shape[-2]
        except:
            return 'N/A'
    else:
        return 'N/A'

def set_seed(seed=1):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def autoregressive_decoding(model, input_ids, max_length=50, temperature=1.0, eos_token_id=None, device=None, use_manual_cache=False):
    """
    Autoregressive generation loop.

    Args:
        model: A PyTorch model with a .forward() method, outputting logits in [batch, seq_len, vocab_size].
        input_ids: Initial input tensor of token ids, shape [batch, seq_len].
        max_length: Maximum total generated length (incl. input).
        temperature: Sampling temperature.
        eos_token_id: Optional, stop generation if this token is generated.
        device: torch.device, if specified.

    Returns:
        Tensor of generated ids, shape [batch, new_seq_len].
    """
    if device is not None:
        input_ids = input_ids.to(device)
        model = model.to(device)
    model.eval()
    generated = input_ids
    past_key_values = None
    
    # Start timing
    start_time = time.perf_counter()
    
    with torch.no_grad():
            for step in range(max_length - input_ids.shape[1]):
                if not use_manual_cache:
                    outputs = model(input_ids=generated, use_cache=False)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    next_token_logits = logits[:, -1, :] / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_tokens], dim=1)
                    if eos_token_id is not None:
                        # stop if all batches have finished
                        if (next_tokens == eos_token_id).all():
                            break
                else:
                    # For first step, process full sequence to build cache
                    # For subsequent steps, only process the new token
                    if past_key_values is None:
                        # First step: process full initial sequence to build KV cache
                        model_input = generated
                    else:
                        # Subsequent steps: only process the new token (KV cache has previous tokens)
                        model_input = generated[:, -1:]
                    outputs = model(input_ids=model_input, past_key_values=past_key_values, use_cache=True)
                    # Extract logits and update past_key_values
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    # Update past_key_values for next iteration
                    past_key_values = outputs.past_key_values

                    # Get cache length - handle different cache types
                    # cache_length = get_cache_length(past_key_values)
                    # if cache_length is not None:
                    #     print(f"[Step {step}] cache_length: {cache_length}")
                    next_token_logits = logits[:, -1, :] / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    # Sample next token using same method as non-cached version
                    next_token = torch.multinomial(probs, num_samples=1)
                    # Reshape to [batch, 1] for concatenation (already correct shape from multinomial)
                    # Prepare input_ids for next iteration (only the new token)
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    # Stop if EOS token is generated
                    if eos_token_id is not None:
                        # stop if all batches have finished
                        if (next_token == eos_token_id).all():
                            break
    # End timing
    end_time = time.perf_counter()
    total_time = end_time - start_time
    return generated, total_time


def run_base_caching_experiment( model, input_ids, max_length, tokenizer, temperature=0.9, print_decoded_text=False):
    print(f"Running base caching experiment with model: {model.config._name_or_path}")
    # First generation without cache
    set_seed()
    generated, total_time_no_cache = autoregressive_decoding(model, input_ids, max_length=max_length, temperature=temperature, eos_token_id=tokenizer.eos_token_id, use_manual_cache=False)
    # print(f"Decoded text (no cache): {tokenizer.decode(generated[0], skip_special_tokens=True)}")
    
    # Second generation with cache
    set_seed()
    generated_with_KV_cache, total_time_with_cache = autoregressive_decoding(model, input_ids, max_length=max_length, temperature=temperature, eos_token_id=tokenizer.eos_token_id, use_manual_cache=True)
    # print(f"Decoded text (with cache): {tokenizer.decode(generated_with_KV_cache[0], skip_special_tokens=True)}")
    
    # Calculate and compare timing metrics
    num_generated_tokens_no_cache = generated.shape[1] - input_ids.shape[1]
    num_generated_tokens_with_cache = generated_with_KV_cache.shape[1] - input_ids.shape[1]
    time_per_token_no_cache = total_time_no_cache / num_generated_tokens_no_cache if num_generated_tokens_no_cache > 0 else 0
    time_per_token_with_cache = total_time_with_cache / num_generated_tokens_with_cache if num_generated_tokens_with_cache > 0 else 0
    
    # Check if outputs match
    if torch.equal(generated, generated_with_KV_cache):
        print("\n✓ Outputs match!")
        if print_decoded_text:
            print(f"Decoded text : {tokenizer.decode(generated[0], skip_special_tokens=True)}")
    else:
        print("\n✗ Outputs differ")
        if print_decoded_text:
            print(f"Decoded text (no cache): {tokenizer.decode(generated[0], skip_special_tokens=True)}")
            print(f"Decoded text (with cache): {tokenizer.decode(generated_with_KV_cache[0], skip_special_tokens=True)}")
    print("\n=== Time Comparison ===")
    print(f"Total time (no cache):   {total_time_no_cache:.4f}s")
    print(f"Total time (with cache): {total_time_with_cache:.4f}s")
    print(f"Time per token (no cache):   {time_per_token_no_cache:.4f}s")
    print(f"Time per token (with cache): {time_per_token_with_cache:.4f}s")
    if time_per_token_with_cache > 0:
        speedup = time_per_token_no_cache / time_per_token_with_cache
        print(f"Speedup with KV cache: {speedup:.2f}x")

def compare_model_times(small_model, big_model, max_length, input_ids, eos_token_id, temperature=0.9):
    print(f"Comparing model times for small model: {small_model.config._name_or_path} and big model: {big_model.config._name_or_path}")
    set_seed()
    generated_small_model, total_time_small_model = autoregressive_decoding(small_model, input_ids, max_length=max_length, temperature=temperature, eos_token_id=eos_token_id, use_manual_cache=False)
    num_generated_tokens_small_model = generated_small_model.shape[1] - input_ids.shape[1]
    time_per_token_small_model = total_time_small_model / num_generated_tokens_small_model if num_generated_tokens_small_model > 0 else 0
    set_seed()
    generated_big_model, total_time_big_model = autoregressive_decoding(big_model, input_ids, max_length=max_length, temperature=temperature, eos_token_id=eos_token_id, use_manual_cache=True)
    num_generated_tokens_big_model = generated_big_model.shape[1] - input_ids.shape[1]
    time_per_token_big_model = total_time_big_model / num_generated_tokens_big_model if num_generated_tokens_big_model > 0 else 0
    print(f"Number of generated tokens (small model): {num_generated_tokens_small_model}")
    print(f"Number of generated tokens (big model): {num_generated_tokens_big_model}")
    print(f"Time per token (small model): {time_per_token_small_model:.4f}s")
    print(f"Time per token (big model): {time_per_token_big_model:.4f}s")
    print(f"Slowdown: {time_per_token_big_model / time_per_token_small_model:.2f}x")
    
def main():
    # Set manual seed for reproducibility
    device = torch.device( "mps" if torch.backends.mps.is_available() else "cpu")
    small_model_name = "gpt2"#parameters = 124M
    big_model_name = "gpt2-xl"#parameters = 1.5B
    small_model = AutoModelForCausalLM.from_pretrained(small_model_name).to(device)
    big_model = AutoModelForCausalLM.from_pretrained(big_model_name).to(device)

    tokenizer = AutoTokenizer.from_pretrained(small_model_name)
    input_ids = tokenizer.encode("hey, how are you?", return_tensors="pt").to(device)
    max_length = 150
    temperature = 0.9
    run_base_caching_experiment(small_model, input_ids, max_length, tokenizer, temperature=temperature, print_decoded_text=True)
    compare_model_times(small_model, big_model, max_length, input_ids, tokenizer.eos_token_id, temperature=temperature)
    
if __name__ == "__main__":
    main()



    
