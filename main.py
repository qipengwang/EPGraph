from collections import defaultdict
import json
import pickle
import torch
import sys, os
from pytorch_lightning import seed_everything
import random
import numpy as np
import argparse

from utils import nowstr

sys.path.append(os.getcwd())
from minicpm.modeling_minicpm import MiniCPMModel
from minicpm.configuration_minicpm import MiniCPMConfig


def seed_all(seed: int=0):
    seed_everything(seed)
    torch.manual_seed(0)

    # Set the seed for PyTorch (GPU) if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # For reproducibility
        torch.backends.cudnn.benchmark = False     # Disable cuDNN benchmarking for reproducibility

    # Set the seed for NumPy
    np.random.seed(0)

    # Set the seed for Python's random module
    random.seed(0)


if __name__ == "__main__":
    seed_all(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/minicpm", help="Batch size for benchmarking")
    parser.add_argument("--use_cuda_graph", action="store_true", help="Use CUDA graph for benchmarking", default=False)
    parser.add_argument("--max_capture_size", type=int, default=256, help="Max capture size for CUDA graph")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length for CUDA graph")
    parser.add_argument("--num_hidden_layers", type=int, default=40, help="Max batch size for CUDA graph")
    parser.add_argument("--use_cache", action="store_true", help="Use KVCache", default=False)
    args = parser.parse_args()

    with open("minicpm/config.json", "r") as f:
        d = json.load(f)
    minicpm_config = MiniCPMConfig(**d)
    args_dict = vars(args)  # 将 argparse.Namespace 转为字典
    for key, value in args_dict.items():
        setattr(minicpm_config, key, value)

    minicpm = MiniCPMModel(minicpm_config)  # random parameters
    os.makedirs(args.model_path, exist_ok=True)
    minicpm_config.save_pretrained("models/minicpm")
    minicpm.save_pretrained("models/minicpm")

    minicpm.to(device="cuda", dtype=torch.float16).eval()
    
    try:
        with open(os.path.join(args.model_path, "sequence_list.pt"), "rb") as f:
            sequence_list = pickle.load(f)
    except:
        seed_all(0)
        sequence_list = []
        for seq_len in range(4, args.max_seq_len + 1, 4):
            sequence = torch.randint(10, minicpm_config.vocab_size - 10, (1, seq_len))
            sequence_list.append(sequence)
        with open(os.path.join(args.model_path, "sequence_list.pt"), "wb") as f:
            pickle.dump(sequence_list, f)

    seed_all(0)
    with torch.no_grad():
        seq = sequence_list[-1].to(device="cuda", dtype=torch.int64)
        outputs = minicpm(input_ids=seq, attention_mask=None, return_dict=True)
    
    latencies = defaultdict(list)
    for idx, seq in enumerate(sequence_list):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            seq = seq.to(device="cuda", dtype=torch.int64)
            start_event.record()
            outputs = minicpm(input_ids=seq, attention_mask=None, return_dict=True)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) * 1000  # Convert to micros
            latencies[seq.shape[1]].append(elapsed_time)
            print(f"seq_len: {seq.shape[1]}, latency: {elapsed_time:.3f} us")
    with open(f"latencies-{nowstr()}.json", "w") as f:
        json.dump(latencies, f, indent=4)

