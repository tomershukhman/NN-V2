import torch

def get_device():
    """Get the best available device (CUDA > MPS > CPU) with optimized memory settings"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable memory optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of available GPU memory
        torch.cuda.empty_cache()  # Clear cache before starting
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        # Enable parallel CPU operations if available
        torch.set_num_threads(torch.get_num_threads())
    return device