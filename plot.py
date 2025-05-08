import matplotlib.pyplot as plt
import numpy as np

cur_seq_len = -1
native_latencies = []
graph_latencies = []
seq_lens = []
with open("data.txt") as f:
    for line in f:
        if "Naive Latency" in line:
            native_latencies.append(float(line.strip().split()[-1]))
        elif "Graph" in line:
            graph_latencies.append(float(line.strip().split()[-1]))
        else:
            seq_lens.append(int(line.strip()))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(seq_lens, native_latencies, label="Naive Latency")
ax1.plot(seq_lens, graph_latencies, label="Graph Latency")
ax2.plot(seq_lens, np.array(native_latencies) - np.array(graph_latencies), label="Latency Gain")
ax1.set_xlabel("Sequence Length")
ax2.set_xlabel("Sequence Length")
ax1.set_ylabel("Latency (ms)")
ax2.set_ylabel("Delta Latency (graph - naive)")
ax1.legend()
ax2.legend()
plt.savefig("latency.png")

