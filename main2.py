import numpy as npHz
import matplotlib.pyplot as plt
from brian2 import *

#import brian2cuda
#set_device("cuda_standalone")

np.random.seed(42)

# --------------------------
# Parameters
# --------------------------
neurons_per_word = 10
words = ["cat", "dog", "talk", "walk", "tree", "leaf", "bark", "meow",
         "sky", "blue", "clouds", "sun"]
n_words = len(words)
N_total = n_words * neurons_per_word

neuron_tau = 10*ms
neuron_threshold = 1.0
neuron_reset = 0.0
refractory_period = 2*ms

poisson_baseline_rate = 1*Hz
poisson_word_rate = 80*Hz
duration_cue = 200*ms
duration_assoc = 200*ms

stdp_A_pre = 0.03
stdp_A_post = -0.02
stdp_tau_pre = 20*ms
stdp_tau_post = 20*ms
w_max = 2.0
w_init = 0.05

n_epochs = 50
top_k = 4

# --------------------------
# Associations
# --------------------------
associations = {
    "cat": ["meow", "leaf"],
    "dog": ["bark", "walk"],
    "tree": ["leaf"],
    "sky": ["blue", "clouds", "sun"],
    "talk": ["walk"],
}

word_to_idx = {w: i for i, w in enumerate(words)}

cluster_indices = {
    w: range(i*neurons_per_word, (i+1)*neurons_per_word)
    for i, w in enumerate(words)
}

# --------------------------
# Neurons
# --------------------------
eqs = '''
dv/dt = -v/tau : 1 (unless refractory)
tau : second (constant)
'''

G = NeuronGroup(N_total, eqs,
                threshold='v>{}'.format(neuron_threshold),
                reset='v={}'.format(neuron_reset),
                refractory=refractory_period,
                method='euler')
G.v = 0
G.tau = neuron_tau

# --------------------------
# Per-cluster inhibition
# --------------------------
Inh = NeuronGroup(n_words, 'dv/dt = -v/(5*ms) : 1',
                  threshold='v>1', reset='v=0', method='euler')

S_ei = Synapses(G, Inh, on_pre='v_post += 0.02')  # excite local inhibitor
for i, w in enumerate(words):
    S_ei.connect(i=list(cluster_indices[w]), j=i)

S_ie = Synapses(Inh, G, on_pre='v_post -= 0.5')  # inhibition back to cluster
for i, w in enumerate(words):
    S_ie.connect(i=i, j=list(cluster_indices[w]))

# --------------------------
# Recurrent excitatory with STDP
# --------------------------
model_syn = '''
w : 1
dapre/dt = -apre/stdp_tau_pre : 1 (event-driven)
dapost/dt = -apost/stdp_tau_post : 1 (event-driven)
'''
on_pre = '''
v_post += w
apre += stdp_A_pre
w = clip(w + apost, 0, w_max)
'''
on_post = '''
apost += stdp_A_post
w = clip(w + apre, 0, w_max)
'''

S_rec = Synapses(G, G, model=model_syn, on_pre=on_pre, on_post=on_post)
for wi in range(n_words):
    for wj in range(n_words):
        if wi == wj:
            continue
        S_rec.connect(i=list(cluster_indices[words[wi]]),
                      j=list(cluster_indices[words[wj]]))
S_rec.w = w_init + 0.01*np.random.randn(len(S_rec.w))
S_rec.w = clip(S_rec.w, 0, w_max)

# --------------------------
# Poisson input
# --------------------------
Poissons = PoissonGroup(n_words, rates=poisson_baseline_rate)
P_syn = Synapses(Poissons, G, on_pre='v_post += 0.6')
for idx, w in enumerate(words):
    P_syn.connect(i=idx, j=list(cluster_indices[w]))

# --------------------------
# Monitors
# --------------------------
spikemon = SpikeMonitor(G)

# --------------------------
# Helper: decode top-k
# --------------------------
def decode_topk(spike_counts, k=top_k):
    cluster_sums = []
    for w in words:
        s = np.sum(spike_counts[list(cluster_indices[w])])
        cluster_sums.append((w, s))
    cluster_sums.sort(key=lambda x: x[1], reverse=True)
    return cluster_sums[:k]

# --------------------------
# Training (pairwise sequential)
# --------------------------
train_history = []
pairs = []
for cue, assoc_list in associations.items():
    for a in assoc_list:
        pairs.append((cue, a))

for epoch in range(n_epochs):
    np.random.shuffle(pairs)
    correct = 0
    for cue, assoc in pairs:
        # 1. cue only
        input_rates = np.ones(n_words) * poisson_baseline_rate
        input_rates[word_to_idx[cue]] = poisson_word_rate
        Poissons.rates = input_rates
        run(duration_cue)

        # 2. cue + assoc overlap
        input_rates[word_to_idx[cue]] = poisson_word_rate
        input_rates[word_to_idx[assoc]] = poisson_word_rate
        Poissons.rates = input_rates
        run(100*ms)

        # 3. assoc only
        input_rates[word_to_idx[cue]] = poisson_baseline_rate
        input_rates[word_to_idx[assoc]] = poisson_word_rate
        Poissons.rates = input_rates
        run(200*ms)

    # evaluate retrieval after epoch
    hits = 0
    for cue, assoc in pairs:
        input_rates = np.ones(n_words) * poisson_baseline_rate
        input_rates[word_to_idx[cue]] = poisson_word_rate
        Poissons.rates = input_rates

        spikemon = SpikeMonitor(G)
        run(duration_cue)
        spike_counts = np.bincount(spikemon.i, minlength=N_total)
        topk = decode_topk(spike_counts, k=top_k)
        if assoc in [w for w, s in topk]:
            hits += 1
    acc = hits / len(pairs)
    train_history.append(acc)

    for i in range(N_total):
        total = np.sum(S_rec.w[i, :])
        if total > 0:
            S_rec.w[i, :] = (S_rec.w[i, :] / total) * w_max
            #S_rec.w[i, :] /= total

    print(f"Epoch {epoch+1}/{n_epochs} - pairwise recall acc: {acc:.2f}")

# --------------------------
# Retrieval demo
# --------------------------
print("\n=== Retrieval results ===")
for cue, assoc_list in associations.items():
    input_rates = np.ones(n_words) * poisson_baseline_rate
    input_rates[word_to_idx[cue]] = poisson_word_rate
    Poissons.rates = input_rates

    spikemon = SpikeMonitor(G)
    run(duration_cue)
    spike_counts = np.bincount(spikemon.i, minlength=N_total)
    topk = decode_topk(spike_counts, k=top_k)
    print(f"{cue:6s} -> {[(w, int(s)) for w, s in topk]}")

# --------------------------
# Plot training accuracy
# --------------------------
plt.plot(train_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Pairwise recall accuracy")
plt.ylim(-0.05, 1.05)
plt.show()
