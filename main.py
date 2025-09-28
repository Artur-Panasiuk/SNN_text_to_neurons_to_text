import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import networkx as nx

np.random.seed(42)

# --------------------------
# Parameters
# --------------------------
n_neurons = 12
neurons_per_word = 10
N_total = n_neurons * neurons_per_word  # total neurons

neuron_threshold = 1.0
neuron_reset_value = 0.0
neuron_tau = 10*ms

poisson_baseline_rate = 1.0*Hz

synapse_distance_threshold = 0.5
synapse_weight_decay_factor = 3.0

stdp_a_pre = 0.05
stdp_a_post_scale = 1.0
stdp_w_max = 2.0

duration_per_word = 400*ms
n_epochs = 10
reward_success = 1.0
reward_failure = -0.2

words = ["cat", "dog", "talk", "walk", "tree", "leaf", "bark", "meow"]
word_rates = {
    "cat": 50,
    "dog": 50,
    "talk": 50,
    "walk": 50,
    "tree": 50,
    "leaf": 50,
    "bark": 50,
    "meow": 50
}

# --------------------------
# Word clusters
# --------------------------
word_clusters = {word: range(i, i+1) for i, word in enumerate(words)}  # 1 cluster per word for Poisson
cluster_neuron_indices = {
    word: range(i*neurons_per_word, (i+1)*neurons_per_word) for i, word in enumerate(words)
}

# --------------------------
# Helper: decoder
# --------------------------
def decode_output(spike_counts):
    cluster_sums = {}
    for word, indices in cluster_neuron_indices.items():
        cluster_sums[word] = np.sum(spike_counts[list(indices)])
    return max(cluster_sums, key=cluster_sums.get)

# --------------------------
# Build neurons
# --------------------------
def build_neurons(cluster_positions):
    eqs = '''
    dv/dt = -v/tau : 1
    tau : second (constant)
    x : 1
    y : 1
    '''
    G = NeuronGroup(N_total, eqs,
                    threshold=f'v>{neuron_threshold}',
                    reset=f'v={neuron_reset_value}',
                    method='euler')
    G.tau = neuron_tau
    G.x = np.repeat(cluster_positions[:,0], neurons_per_word)
    G.y = np.repeat(cluster_positions[:,1], neurons_per_word)
    return G

# --------------------------
# Build plastic synapses
# --------------------------
def build_plastic_synapses(G, cluster_positions):
    A_pre = stdp_a_pre
    A_post = -A_pre*stdp_a_post_scale

    model_syn = '''
    w : 1
    reward : 1 (shared)
    dapre/dt = -apre/(20*ms) : 1 (event-driven)
    dapost/dt = -apost/(20*ms) : 1 (event-driven)
    '''

    on_pre = f'''
    v_post += w
    apre += {A_pre}
    w = clip(w + reward*apost, 0, {stdp_w_max})
    '''

    on_post = f'''
    apost += {A_post}
    w = clip(w + reward*apre, 0, {stdp_w_max})
    '''

    S = Synapses(G, G, model=model_syn, on_pre=on_pre, on_post=on_post)

    # Connect clusters to themselves and nearby clusters
    for i in range(n_neurons):
        print(f"{i} of {n_neurons}")
        si = slice(i*neurons_per_word, (i+1)*neurons_per_word)
        for j in range(n_neurons):
            if i==j: continue
            sj = slice(j*neurons_per_word, (j+1)*neurons_per_word)
            d = np.linalg.norm(cluster_positions[i]-cluster_positions[j])
            if d < synapse_distance_threshold:
                S.connect(i=range(si.start, si.stop), j=range(sj.start, sj.stop))
                S.w[si.start:si.stop, sj.start:sj.stop] = 0.1*np.exp(-d*synapse_weight_decay_factor)

    S.reward = 0.0
    return S

# --------------------------
# Cluster positions
# --------------------------
cluster_positions = np.random.rand(n_neurons,2)

# --------------------------
# Neurons, synapses, Poisson inputs
# --------------------------
Neu = build_neurons(cluster_positions)
Syn = build_plastic_synapses(Neu, cluster_positions)
Poisson_inputs = PoissonGroup(len(words), rates=[poisson_baseline_rate]*len(words))

# Connect Poisson neurons to corresponding clusters
Poisson_syn = Synapses(Poisson_inputs, Neu, on_pre='v_post += 0.5')
for idx, word in enumerate(words):
    target_neurons = list(cluster_neuron_indices[word])
    Poisson_syn.connect(i=idx, j=target_neurons)

# --------------------------
# Monitors
# --------------------------
M_spikes = SpikeMonitor(Neu)
M_poisson = SpikeMonitor(Poisson_inputs)

# --------------------------
# Training loop
# --------------------------
train_history = []

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}/{n_epochs}")

    correct = 0
    total = 0

    for target_idx, word in enumerate(words):
        print(f" Presenting: {word}")

        Poisson_inputs.rates[:] = poisson_baseline_rate
        Poisson_inputs.rates[target_idx] = word_rates[word]*Hz

        Syn.reward = reward_success

        M_spikes = SpikeMonitor(Neu)
        M_poisson = SpikeMonitor(Poisson_inputs)

        run(duration_per_word)

        spike_counts = np.bincount(M_spikes.i, minlength=N_total)
        predicted = decode_output(spike_counts)

        if predicted == word:
            print(f"  ✔ Correct: {predicted}")
            correct += 1
        else:
            print(f"  ✘ Wrong: {predicted}")
            Syn.reward = reward_failure

        total += 1

    acc = correct / total
    train_history.append(acc)
    print(f"Epoch accuracy: {acc:.2f}")

# --------------------------
# Plot learning curve
# --------------------------
plt.figure()
plt.plot(train_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Learning curve")
plt.show()

# --------------------------
# Visualization: original positions
# --------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
colors = np.zeros(N_total)
for idx, (word, indices) in enumerate(cluster_neuron_indices.items()):
    colors[list(indices)] = idx
plt.scatter(Neu.x[:], Neu.y[:], c=colors, cmap="tab20", s=50, edgecolors='black')
plt.title("Original cluster positions")
plt.xlabel("x")
plt.ylabel("y")

# --------------------------
# Visualization: functional clustering
# --------------------------
plt.subplot(1,2,2)
G_plot = nx.Graph()
weights = np.array(Syn.w[:]).flatten()
for i,j,w in zip(Syn.i, Syn.j, weights):
    G_plot.add_edge(i,j,weight=w)
pos = nx.spring_layout(G_plot, weight='weight', seed=42)
nx.draw_networkx_nodes(G_plot, pos, node_color=colors, cmap="tab20", node_size=50)
nx.draw_networkx_edges(G_plot, pos, alpha=0.3)
plt.title("Functional clustering (force-directed)")
plt.axis('off')
plt.tight_layout()
plt.show()
