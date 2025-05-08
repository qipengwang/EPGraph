import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import json

sys.path.append(os.getcwd())
from deepseek.modeling_deepseek import DeepseekV3MLP, DeepseekV3Attention
from deepseek.configuration_deepseek import DeepseekV3Config

@torch.no_grad()
def test(seq_len):
# seq_len = 128
    warmup_iter, run_iter = 10, 100
    with open("deepseek/config.json", "r") as f:
        d = json.load(f)
    config = DeepseekV3Config(**d)
    x = torch.randn(1, seq_len, config.hidden_size).cuda()
    w = torch.randn(1, seq_len, 1).cuda()
    attn = DeepseekV3Attention(config).cuda()
    mlp = DeepseekV3MLP(config).cuda()

    for _ in range(warmup_iter):
        hid_state, _, _ = attn(x)
        hid_state = torch.mul(mlp(hid_state), w)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(run_iter):
        hid_state, _, _ = attn(x)
        hid_state = torch.mul(mlp(hid_state), w)
    end_event.record()
    torch.cuda.synchronize()
    print("Naive Latency: ", start_event.elapsed_time(end_event)) # in ms
    naive_latency = start_event.elapsed_time(end_event)  # in ms

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hid_state, _, _ = attn(x)
        hid_state = torch.mul(mlp(hid_state), w)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(run_iter):
        graph.replay()
    end_event.record()
    torch.cuda.synchronize()
    print("Graph Latency: ", start_event.elapsed_time(end_event)) # in ms
    graph_latency = start_event.elapsed_time(end_event)  # in ms

    return naive_latency, graph_latency


if __name__ == "__main__":
    test(16)
    test(32)
    test(64)
    test(128)
    test(256)
    test(512)
    test(1024)
    naive_latencies, graph_latencies = [], []
    seq_len_list = list(range(16, 1025, 16))
    for seq_len in seq_len_list:
        print(seq_len)
        naive_latency, graph_latency = test(seq_len)
        naive_latencies.append(naive_latency)
        graph_latencies.append(graph_latency)

    plt.plot(seq_len_list, naive_latencies, label="Naive Latency")
    plt.plot(seq_len_list, graph_latencies, label="Graph Latency")
    plt.xlabel("Sequence Length")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Sequence Length")
    plt.legend()
    plt.savefig("latency-seqlen.png")

        
