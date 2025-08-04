import random
import numpy as np
import torch



def setseed(Random_seed):
    random.seed(Random_seed)
    np.random.seed(Random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Random_seed)

    torch.manual_seed(Random_seed)
    torch.cuda.manual_seed(Random_seed)
    torch.cuda.manual_seed_all(Random_seed)


def set_llm_models(model_path: str):

    # set "select_model" to your llm file path

    global hidden_dim, select_model

    attn_implementation = "flash_attention_2" # you can switch to "eager" or "flash_attention_2" depending on your hardware and PyTorch version

    if model_path == "gpt2":
        hidden_dim = 768
        select_model = "gpt2"

    elif model_path == "llama3.2-1B":
        hidden_dim = 2048
        select_model = "your_model_path"

    elif model_path == "DeepSeek-R1-Distill-Qwen-1.5B":
        hidden_dim = 1536
        select_model = "your_model_path"

    elif model_path == "gemma-3-1b":
        hidden_dim = 1152
        select_model = "your_model_path"
        attn_implementation = "eager"

    elif model_path == "SmolLM2-135M":
        hidden_dim = 576
        select_model = "your_model_path"

    elif model_path == "Qwen2.5-0.5B":
        hidden_dim = 896
        select_model = "your_model_path"

    # you can add more LLMs here as long as your GPU can handle them


    return hidden_dim, select_model, attn_implementation
