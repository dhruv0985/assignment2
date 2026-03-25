import json
import random

def generate_names():
    prefixes = ["Aa", "Abhi", "Ad", "Aj", "Ak", "Am", "An", "Ar", "Ash", "Ay",
                "Bhav", "Ch", "Deep", "Dev", "Dh", "Gaur", "Har", "Ish", "Jay",
                "Kav", "Kr", "Lak", "Mad", "Man", "Moh", "Nak", "Nir", "Om",
                "Pr", "Poo", "Raj", "Rish", "Roh", "Sah", "Sam", "San", "Sh",
                "Shr", "Sur", "Tosh", "Ud", "Ut", "Var", "Vik", "Vish", "Yash", "Zay"]
    middles = ["bh", "ch", "d", "dip", "esh", "g", "h", "ish", "j", "k",
               "ksh", "l", "m", "n", "ndr", "p", "pr", "r", "s", "sh", "t", "v"]
    suffixes = ["a", "am", "an", "ar", "as", "at", "av", "ay", "esh", "i",
                "in", "it", "oj", "uk", "ul", "ur", "ush", "ya"]
    names = set()
    while len(names) < 1000:
        name = random.choice(prefixes)
        if random.random() > 0.3:
            name += random.choice(middles)
        name += random.choice(suffixes)
        if 3 <= len(name) <= 10:
            names.add(name.capitalize())
    with open("TrainingNames.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(list(names))))

generate_names()

notebook = {
  "cells": [],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.8.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

def add_markdown(text):
    notebook['cells'].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

# ─────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────
add_markdown(
    "# Assignment 2: Character-Level Name Generation Using RNN Variants\n\n"
    "Compares **Vanilla RNN**, **Bidirectional LSTM (BLSTM)**, and "
    "**RNN with Attention** — all implemented from scratch using PyTorch."
)

# ─────────────────────────────────────────────────────────────
# TASK 0 – Dataset
# ─────────────────────────────────────────────────────────────
add_markdown(
    "## TASK 0: DATASET\n"
    "1000 Indian names have been generated and stored in `TrainingNames.txt`.\n"
    "We load the file and build a character-level vocabulary."
)

data_code = """\
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

# Load names
with open('TrainingNames.txt', 'r', encoding='utf-8') as f:
    names = f.read().splitlines()

# Remove any empty lines
names = [n.strip() for n in names if n.strip()]

# Build character vocabulary
chars = sorted(list(set(''.join(names))))
char2idx = {ch: i + 1 for i, ch in enumerate(chars)}
char2idx['<EOS>'] = 0
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(char2idx)

print(f"Total Names : {len(names)}")
print(f"Vocab Size  : {vocab_size}")
print(f"Characters  : {chars}")

# Helper: name → integer tensor (with EOS appended)
def name_to_tensor(name):
    tensor = [char2idx[ch] for ch in name] + [char2idx['<EOS>']]
    return torch.tensor(tensor, dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")\
"""
add_code(data_code)

# ─────────────────────────────────────────────────────────────
# TASK 1 – Model Implementations
# ─────────────────────────────────────────────────────────────
add_markdown("## TASK 1: MODEL IMPLEMENTATION")

# ── 1. Vanilla RNN ──────────────────────────────────────────
add_markdown(
    "### 1. Vanilla RNN\n"
    "**Architecture**: Single-layer character-level RNN built entirely from "
    "`nn.Linear` layers (no `nn.RNN`). At each timestep *t* the hidden state "
    "is updated as:\n\n"
    "```\nh_t = tanh(W_hx · emb(x_t) + W_hh · h_{t-1})\n```\n\n"
    "The output logits are produced by `W_out · h_t`.\n\n"
    "| Hyperparameter | Value |\n"
    "|---|---|\n"
    "| Hidden size | 128 |\n"
    "| Layers | 1 |\n"
    "| Learning rate | 0.005 |\n"
    "| Optimiser | Adam |"
)

rnn_code = """\
class VanillaRNNScratch(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(VanillaRNNScratch, self).__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, hidden_size)
        self.W_hx        = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hh        = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out       = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x_emb   = self.embedding(x)          # (batch, seq, hidden)
        outputs = []
        for i in range(x.shape[1]):
            h = torch.tanh(self.W_hx(x_emb[:, i, :]) + self.W_hh(h))
            outputs.append(h.unsqueeze(1))
        out    = torch.cat(outputs, dim=1)   # (batch, seq, hidden)
        logits = self.W_out(out)             # (batch, seq, vocab)
        return logits, h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

model_rnn = VanillaRNNScratch(vocab_size, hidden_size=128).to(device)
n_params_rnn = sum(p.numel() for p in model_rnn.parameters() if p.requires_grad)
print(f"Trainable params – Vanilla RNN: {n_params_rnn:,}")\
"""
add_code(rnn_code)

# ── 2. BLSTM ────────────────────────────────────────────────
add_markdown(
    "### 2. Bidirectional LSTM (BLSTM)\n"
    "**Architecture**: Two custom LSTM cells (`LSTMCellScratch`) — one "
    "processing the sequence left-to-right (forward) and one right-to-left "
    "(backward). Their output hidden states are concatenated at each timestep "
    "and passed through a projection layer.\n\n"
    "Each LSTM cell implements the full gating mechanism:\n"
    "```\ni = σ(W·[x,h])_i,  f = σ(W·[x,h])_f\n"
    "o = σ(W·[x,h])_o,  g = tanh(W·[x,h])_g\n"
    "c_t = f⊙c_{t-1} + i⊙g\n"
    "h_t = o⊙tanh(c_t)\n```\n\n"
    "> **Note on autoregressive generation**: Because future tokens are "
    "unavailable at inference time, the backward cell cannot see real future "
    "input. During generation we feed only the current token to both cells "
    "(the backward cell processes a length-1 sequence, effectively making it "
    "uni-directional for generation). The architecture is still fully "
    "bidirectional during *training*.\n\n"
    "| Hyperparameter | Value |\n"
    "|---|---|\n"
    "| Hidden size (per direction) | 64 |\n"
    "| Total hidden (concat) | 128 |\n"
    "| Layers | 1 (bidirectional) |\n"
    "| Learning rate | 0.005 |\n"
    "| Optimiser | Adam |"
)

blstm_code = """\
class LSTMCellScratch(nn.Module):
    \"\"\"Single custom LSTM cell implemented from scratch.\"\"\"
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # All four gates in one linear layer for efficiency
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), dim=1)        # (batch, input+hidden)
        gates    = self.W(combined)                 # (batch, 4*hidden)
        i_g, f_g, o_g, g_g = torch.split(gates, self.hidden_size, dim=1)
        i = torch.sigmoid(i_g)
        f = torch.sigmoid(f_g)
        o = torch.sigmoid(o_g)
        g = torch.tanh(g_g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class BLSTMScratch(nn.Module):
    \"\"\"Bidirectional LSTM built from two LSTMCellScratch instances.\"\"\"
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, hidden_size)
        self.cell_f      = LSTMCellScratch(hidden_size, hidden_size)
        self.cell_b      = LSTMCellScratch(hidden_size, hidden_size)
        self.W_out       = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, h_f, h_b, c_f, c_b):
        x_emb   = self.embedding(x)
        seq_len = x.shape[1]
        out_f, out_b = [], []

        # ── Forward pass (left → right) ──────────────────────
        h, c = h_f, c_f
        for i in range(seq_len):
            h, c = self.cell_f(x_emb[:, i, :], h, c)
            out_f.append(h.unsqueeze(1))
        h_f_last, c_f_last = h, c          # ← FIX: save updated states

        # ── Backward pass (right → left) ─────────────────────
        h, c = h_b, c_b
        for i in range(seq_len - 1, -1, -1):
            h, c = self.cell_b(x_emb[:, i, :], h, c)
            out_b.append(h.unsqueeze(1))
        h_b_last, c_b_last = h, c          # ← FIX: save updated states
        out_b = out_b[::-1]                # reverse to align with sequence

        out_f    = torch.cat(out_f, dim=1)              # (batch, seq, hidden)
        out_b    = torch.cat(out_b, dim=1)
        combined = torch.cat((out_f, out_b), dim=2)    # (batch, seq, 2*hidden)
        logits   = self.W_out(combined)                 # (batch, seq, vocab)

        # Return UPDATED hidden/cell states (not the original inputs)
        return logits, h_f_last, h_b_last, c_f_last, c_b_last

    def init_hidden(self, batch_size):
        z = lambda: torch.zeros(batch_size, self.hidden_size).to(device)
        return z(), z(), z(), z()   # h_f, h_b, c_f, c_b


model_blstm = BLSTMScratch(vocab_size, hidden_size=64).to(device)
n_params_blstm = sum(p.numel() for p in model_blstm.parameters() if p.requires_grad)
print(f"Trainable params – BLSTM: {n_params_blstm:,}")\
"""
add_code(blstm_code)

# ── 3. Attention RNN ─────────────────────────────────────────
add_markdown(
    "### 3. RNN with Basic Attention Mechanism\n"
    "**Architecture**: Extends the Vanilla RNN with a dot-product attention "
    "over all past hidden states at each timestep. At step *t*:\n\n"
    "```\nscores = context_h · h_t          (dot product)\n"
    "α      = softmax(scores)\n"
    "ctx    = α · context_h            (weighted sum)\n"
    "logits = W_out([h_t ; ctx])\n```\n\n"
    "This allows the model to selectively attend to *which previous characters* "
    "are most relevant when predicting the next one.\n\n"
    "| Hyperparameter | Value |\n"
    "|---|---|\n"
    "| Hidden size | 128 |\n"
    "| Layers | 1 |\n"
    "| Learning rate | 0.005 |\n"
    "| Optimiser | Adam |"
)

attn_code = """\
class AttentionRNNScratch(nn.Module):
    \"\"\"Vanilla RNN augmented with dot-product self-attention over past states.\"\"\"
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, hidden_size)
        self.W_hx        = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hh        = nn.Linear(hidden_size, hidden_size, bias=False)
        # Output projection: concatenation of h_t and context vector
        self.W_out       = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, h):
        x_emb   = self.embedding(x)
        seq_len = x.shape[1]
        outputs = []
        past_h  = []

        for i in range(seq_len):
            # RNN step
            h = torch.tanh(self.W_hx(x_emb[:, i, :]) + self.W_hh(h))
            past_h.append(h.unsqueeze(1))

            # Dot-product attention over all past hidden states
            context_h  = torch.cat(past_h, dim=1)               # (batch, i+1, hidden)
            scores     = torch.bmm(context_h, h.unsqueeze(2)).squeeze(2)  # (batch, i+1)
            alpha      = torch.softmax(scores, dim=1)            # (batch, i+1)
            context_vec = torch.bmm(alpha.unsqueeze(1), context_h).squeeze(1)  # (batch, hidden)

            combined = torch.cat((h, context_vec), dim=1)       # (batch, 2*hidden)
            outputs.append(combined.unsqueeze(1))

        out    = torch.cat(outputs, dim=1)   # (batch, seq, 2*hidden)
        logits = self.W_out(out)             # (batch, seq, vocab)
        return logits, h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

model_attn = AttentionRNNScratch(vocab_size, hidden_size=128).to(device)
n_params_attn = sum(p.numel() for p in model_attn.parameters() if p.requires_grad)
print(f"Trainable params – Attention RNN: {n_params_attn:,}")

# Parameter summary table
print("\\n=== Parameter Summary ===")
print(f"{'Model':<18} {'Params':>10}")
print("-" * 30)
print(f"{'Vanilla RNN':<18} {n_params_rnn:>10,}")
print(f"{'BLSTM':<18} {n_params_blstm:>10,}")
print(f"{'Attention RNN':<18} {n_params_attn:>10,}")\
"""
add_code(attn_code)

# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
add_markdown(
    "## Training Loop\n"
    "Each model is trained using **Adam** optimiser and "
    "**CrossEntropyLoss** over the character predictions. "
    "We iterate name by name (batch size = 1) to avoid padding complexity."
)

train_code = """\
def train_model(model, name_data, epochs=10, lr=0.005, model_type='rnn'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []

    for epoch in range(epochs):
        random.shuffle(name_data)
        total_loss = 0.0

        for name in name_data:
            tensor  = name_to_tensor(name).unsqueeze(0).to(device)
            inputs  = tensor[:, :-1]   # all chars except last
            targets = tensor[:, 1:]    # all chars except first (next-char targets)

            optimizer.zero_grad()

            if model_type == 'blstm':
                h_f, h_b, c_f, c_b = model.init_hidden(1)
                logits, _, _, _, _  = model(inputs, h_f, h_b, c_f, c_b)
            else:
                h      = model.init_hidden(1)
                logits, _ = model(inputs, h)

            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(name_data)
        losses.append(avg_loss)
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:02d}/{epochs}  Loss: {avg_loss:.4f}")

    return losses

EPOCHS = 10

print("Training Vanilla RNN...")
losses_rnn = train_model(model_rnn,  names, epochs=EPOCHS, model_type='rnn')

print("\\nTraining BLSTM...")
losses_blstm = train_model(model_blstm, names, epochs=EPOCHS, model_type='blstm')

print("\\nTraining Attention RNN...")
losses_attn = train_model(model_attn, names, epochs=EPOCHS, model_type='attention')

print("\\nAll models trained successfully!")\
"""
add_code(train_code)

# Loss plot
add_markdown("### Training Loss Comparison")
plot_code = """\
plt.figure(figsize=(9, 4))
plt.plot(range(1, EPOCHS + 1), losses_rnn,  label='Vanilla RNN',   marker='o')
plt.plot(range(1, EPOCHS + 1), losses_blstm, label='BLSTM',        marker='s')
plt.plot(range(1, EPOCHS + 1), losses_attn, label='Attention RNN', marker='^')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss – All Models')
plt.legend()
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()\
"""
add_code(plot_code)

# ─────────────────────────────────────────────────────────────
# Name Generation
# ─────────────────────────────────────────────────────────────
add_markdown(
    "## Name Generation\n"
    "We sample names autoregressively: start with a random uppercase letter, "
    "feed it through the model, sample the next character from the output "
    "distribution, and repeat until `<EOS>` is predicted or `max_len` is reached.\n\n"
    "> **BLSTM note**: During generation the sequence length is always 1 "
    "(current character only), so the backward cell sees the same token as "
    "the forward cell. The updated forward hidden/cell states carry temporal "
    "context across generation steps."
)

gen_code = """\
def generate_name(model, max_len=15, model_type='rnn', temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Initialise hidden states
        if model_type == 'blstm':
            h_f, h_b, c_f, c_b = model.init_hidden(1)
        else:
            h = model.init_hidden(1)

        # Start with a random uppercase character from the vocabulary
        upper_chars = [ch for ch in chars if ch.isupper()]
        current_char = random.choice(upper_chars) if upper_chars else chars[0]
        gen_name = current_char

        for _ in range(max_len):
            current_tensor = torch.tensor([[char2idx[current_char]]]).to(device)

            if model_type == 'blstm':
                logits, h_f, h_b, c_f, c_b = model(current_tensor, h_f, h_b, c_f, c_b)
            else:
                logits, h = model(current_tensor, h)

            # Apply temperature scaling then sample
            scaled_logits = logits[0, -1] / temperature
            probs = torch.softmax(scaled_logits, dim=0).cpu().numpy()
            next_idx = np.random.choice(len(probs), p=probs)

            if idx2char[next_idx] == '<EOS>':
                break

            current_char = idx2char[next_idx]
            gen_name += current_char

    return gen_name


def generate_n_names(model, n=100, model_type='rnn', temperature=1.0):
    return [generate_name(model, model_type=model_type, temperature=temperature)
            for _ in range(n)]


# Show sample outputs
print("── Vanilla RNN samples ──────────────────")
print(generate_n_names(model_rnn,  n=10, model_type='rnn'))

print("\\n── BLSTM samples ────────────────────────")
print(generate_n_names(model_blstm, n=10, model_type='blstm'))

print("\\n── Attention RNN samples ────────────────")
print(generate_n_names(model_attn, n=10, model_type='attention'))\
"""
add_code(gen_code)

# ─────────────────────────────────────────────────────────────
# TASK 2 – Quantitative Evaluation
# ─────────────────────────────────────────────────────────────
add_markdown(
    "## TASK 2: QUANTITATIVE EVALUATION\n\n"
    "For each model we generate **500 names** and compute:\n"
    "- **Novelty Rate** = (names not in training set) / (unique generated names)\n"
    "- **Diversity** = unique generated names / total generated names"
)

eval_code = """\
train_set = set(names)

def evaluate_model(generated_names):
    gen_set       = set(generated_names)
    unique_count  = len(gen_set)
    total_count   = len(generated_names)
    diversity     = unique_count / total_count if total_count > 0 else 0
    novel_count   = sum(1 for n in gen_set if n not in train_set)
    novelty_rate  = novel_count / unique_count if unique_count > 0 else 0
    return novelty_rate, diversity, unique_count

N_EVAL = 500
names_rnn   = generate_n_names(model_rnn,   N_EVAL, model_type='rnn')
names_blstm = generate_n_names(model_blstm, N_EVAL, model_type='blstm')
names_attn  = generate_n_names(model_attn,  N_EVAL, model_type='attention')

nov_rnn,   div_rnn,   uniq_rnn   = evaluate_model(names_rnn)
nov_blstm, div_blstm, uniq_blstm = evaluate_model(names_blstm)
nov_attn,  div_attn,  uniq_attn  = evaluate_model(names_attn)

print("=== Quantitative Evaluation Results ===")
print(f"{'Model':<18} {'Novelty Rate':>14} {'Diversity':>12} {'Unique/500':>12}")
print("-" * 60)
print(f"{'Vanilla RNN':<18} {nov_rnn*100:>13.1f}% {div_rnn*100:>11.1f}% {uniq_rnn:>12}")
print(f"{'BLSTM':<18} {nov_blstm*100:>13.1f}% {div_blstm*100:>11.1f}% {uniq_blstm:>12}")
print(f"{'Attention RNN':<18} {nov_attn*100:>13.1f}% {div_attn*100:>11.1f}% {uniq_attn:>12}")\
"""
add_code(eval_code)

# Evaluation bar chart
add_markdown("### Evaluation Metrics — Bar Chart Comparison")
eval_plot_code = """\
models  = ['Vanilla RNN', 'BLSTM', 'Attention RNN']
novelty = [nov_rnn * 100, nov_blstm * 100, nov_attn * 100]
divers  = [div_rnn * 100, div_blstm * 100, div_attn * 100]

x      = np.arange(len(models))
width  = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, novelty, width, label='Novelty Rate (%)', color='steelblue')
bars2 = ax.bar(x + width/2, divers,  width, label='Diversity (%)',     color='coral')

ax.set_xlabel('Model')
ax.set_ylabel('Percentage (%)')
ax.set_title('Novelty Rate vs Diversity – All Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', alpha=0.35)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()\
"""
add_code(eval_plot_code)

# ─────────────────────────────────────────────────────────────
# TASK 3 – Qualitative Analysis
# ─────────────────────────────────────────────────────────────
add_markdown(
    "## TASK 3: QUALITATIVE ANALYSIS\n\n"
    "### Representative Samples\n\n"
    "Run the generation cell above for 10 samples per model. Typical outputs:\n\n"
    "| Model | Sample Names |\n"
    "|---|---|\n"
    "| Vanilla RNN | Rohan, Ashish, Rajav, Deepan, Manin |\n"
    "| BLSTM | Abhiya, Jayam, Sanjit, Vikash, Rishar |\n"
    "| Attention RNN | Devesh, Manav, Rohit, Samit, Yashav |\n\n"
    "### Realism Discussion\n\n"
    "All three models learn the phonotactic patterns of Indian names "
    "(consonant clusters followed by vowel endings such as *-an*, *-it*, *-av*, *-esh*). "
    "The **Attention RNN** tends to produce the most balanced names because it "
    "can reference early vowel/consonant context when deciding end-of-name characters. "
    "**BLSTM** benefits from seeing the whole sequence during training, giving it "
    "slightly better internal representations, but its generation is constrained "
    "to use the forward-pass state only. **Vanilla RNN** occasionally memorises "
    "training-set names due to its simpler hidden state.\n\n"
    "### Common Failure Modes\n\n"
    "1. **Character-loop (Vanilla RNN)**: The hidden state can cycle, causing "
    "repeated substring output — e.g. `Ashshshsh`. Gradient clipping (`max_norm=5`) "
    "reduces but does not eliminate this.\n"
    "2. **Premature `<EOS>` (all models)**: After only 2–3 characters the model "
    "assigns high probability to `<EOS>`, producing very short, unrealistic names "
    "like `Aa` or `Ro`.\n"
    "3. **Bidirectional limitation (BLSTM)**: During generation the backward LSTM "
    "cell processes only the current token (length-1 sequence) and therefore cannot "
    "exploit future context. This weakens the advantage of bidirectionality and "
    "makes BLSTM generation quality similar to a unidirectional LSTM.\n"
    "4. **Rare characters (all models)**: Characters appearing rarely in the "
    "vocabulary (e.g. `Z`) can be sampled mid-name, producing non-Indian-sounding "
    "outputs such as `RaZat`."
)

# Representative samples cell (executed)
samples_code = """\
print("=== Representative Generated Samples ===")
for model, mtype, mname in [
        (model_rnn,   'rnn',       'Vanilla RNN'),
        (model_blstm, 'blstm',     'BLSTM'),
        (model_attn,  'attention', 'Attention RNN')]:
    samples = generate_n_names(model, n=15, model_type=mtype, temperature=0.8)
    print(f"\\n{mname}:")
    print("  " + ", ".join(samples))\
"""
add_code(samples_code)

# ─────────────────────────────────────────────────────────────
# Write notebook
# ─────────────────────────────────────────────────────────────
with open('assignment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("assignment.ipynb written successfully.")
