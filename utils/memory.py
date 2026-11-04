import gc
import torch
import psutil


def print_memory_usage():
    if torch.cuda.is_available():
        print(f"{torch.cuda.memory_allocated()/ 1024 ** 3:.2f} GB allocated")
        print(f"{torch.cuda.memory_reserved()/ 1024 ** 3:.2f} GB reserved")

        print(
            f"torch memory summary: {torch.cuda.memory_summary(device=None, abbreviated=None)}"
        )
    process = psutil.Process()
    mem_info = process.memory_info()

    print(f"System RAM: {mem_info.rss / 1024**3:.2f}GB used by process")
    print(f"System RAM: {psutil.virtual_memory().percent:.1f}% total system usage")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    else:
        try:
            torch._dynamo.reset()
        except Exception as e:
            print(f"Could not reset dynamo: {e}")
