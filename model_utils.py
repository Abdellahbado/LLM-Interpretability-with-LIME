import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", message=".*Using pad_token_id.*")

def load_model_and_tokenizer(model_name: str):
    """
    Loads the specified causal LM model and tokenizer, and sets the device.
    Prioritizes MPS (for macOS) > CPU.

    Args:
        model_name: The name of the model to load from Hugging Face Hub.

    Returns:
        A tuple containing (model, tokenizer, device).
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("MPS (or CUDA) not available. Using CPU.")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)
    print(f"Model moved to {device}.")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            print("Set pad_token to eos_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            print("Added a new [PAD] token as pad_token.")


    return model, tokenizer, device
