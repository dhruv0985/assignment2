import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 1. Load data
training_file = r'c:\Users\Asus\Downloads\assignment\p2\TrainingNames.txt'
with open(training_file, 'r', encoding='utf-8') as f:
    names = [n.strip() for n in f.read().splitlines() if n.strip()]

chars = sorted(list(set(''.join(names))))
char2idx = {ch: i + 1 for i, ch in enumerate(chars)}
char2idx['<EOS>'] = 0
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(char2idx)
train_set = set(names)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def name_to_tensor(name):
    tensor = [char2idx[ch] for ch in name] + [char2idx['<EOS>']]
    return torch.tensor(tensor, dtype=torch.long)

# 2. Model Definitions
class VanillaRNNScratch(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(VanillaRNNScratch, self).__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, hidden_size)
        self.W_hx        = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hh        = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out       = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x_emb   = self.embedding(x)
        outputs = []
        for i in range(x.shape[1]):
            h = torch.tanh(self.W_hx(x_emb[:, i, :]) + self.W_hh(h))
            outputs.append(h.unsqueeze(1))
        out    = torch.cat(outputs, dim=1)
        logits = self.W_out(out)
        return logits, h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

class LSTMCellScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, h, c):
        combined = torch.cat((x, h), dim=1)
        gates    = self.W(combined)
        i_g, f_g, o_g, g_g = torch.split(gates, self.hidden_size, dim=1)
        i = torch.sigmoid(i_g)
        f = torch.sigmoid(f_g)
        o = torch.sigmoid(o_g)
        g = torch.tanh(g_g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class BLSTMScratch(nn.Module):
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
        
        # Forward
        h, c = h_f, c_f
        for i in range(seq_len):
            h, c = self.cell_f(x_emb[:, i, :], h, c)
            out_f.append(h.unsqueeze(1))
        h_f_last, c_f_last = h, c

        # Backward
        h, c = h_b, c_b
        for i in range(seq_len - 1, -1, -1):
            h, c = self.cell_b(x_emb[:, i, :], h, c)
            out_b.append(h.unsqueeze(1))
        h_b_last, c_b_last = h, c
        out_b = out_b[::-1]

        out_f    = torch.cat(out_f, dim=1)
        out_b    = torch.cat(out_b, dim=1)
        combined = torch.cat((out_f, out_b), dim=2)
        logits   = self.W_out(combined)
        return logits, h_f_last, h_b_last, c_f_last, c_b_last

    def init_hidden(self, batch_size):
        z = lambda: torch.zeros(batch_size, self.hidden_size).to(device)
        return z(), z(), z(), z()

class AttentionRNNScratch(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(vocab_size, hidden_size)
        self.W_hx        = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hh        = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out       = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, h):
        x_emb   = self.embedding(x)
        seq_len = x.shape[1]
        outputs = []
        past_h  = []
        for i in range(seq_len):
            h = torch.tanh(self.W_hx(x_emb[:, i, :]) + self.W_hh(h))
            past_h.append(h.unsqueeze(1))
            context_h  = torch.cat(past_h, dim=1)
            scores     = torch.bmm(context_h, h.unsqueeze(2)).squeeze(2)
            alpha      = torch.softmax(scores, dim=1)
            context_vec = torch.bmm(alpha.unsqueeze(1), context_h).squeeze(1)
            combined = torch.cat((h, context_vec), dim=1)
            outputs.append(combined.unsqueeze(1))
        out    = torch.cat(outputs, dim=1)
        logits = self.W_out(out)
        return logits, h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

# 3. Training Function
def train_model(model, name_data, epochs=5, model_type='rnn'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for _ in range(epochs):
        random.shuffle(name_data)
        for name in name_data:
            tensor  = name_to_tensor(name).unsqueeze(0).to(device)
            inputs  = tensor[:, :-1]
            targets = tensor[:, 1:]
            optimizer.zero_grad()
            if model_type == 'blstm':
                h_f, h_b, c_f, c_b = model.init_hidden(1)
                logits, _, _, _, _  = model(inputs, h_f, h_b, c_f, c_b)
            else:
                h = model.init_hidden(1)
                logits, _ = model(inputs, h)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

# 4. Generation Functions
def generate_name(model, model_type='rnn', max_len=15):
    model.eval()
    with torch.no_grad():
        if model_type == 'blstm':
            h_f, h_b, c_f, c_b = model.init_hidden(1)
        else:
            h = model.init_hidden(1)
        upper_chars = [ch for ch in chars if ch.isupper()]
        current_char = random.choice(upper_chars) if upper_chars else chars[0]
        gen_name = current_char
        for _ in range(max_len):
            current_tensor = torch.tensor([[char2idx[current_char]]]).to(device)
            if model_type == 'blstm':
                logits, h_f, h_b, c_f, c_b = model(current_tensor, h_f, h_b, c_f, c_b)
            else:
                logits, h = model(current_tensor, h)
            probs = torch.softmax(logits[0, -1], dim=0).cpu().numpy()
            next_idx = np.random.choice(len(probs), p=probs)
            if idx2char[next_idx] == '<EOS>':
                break
            current_char = idx2char[next_idx]
            gen_name += current_char
    return gen_name

def generate_n_names(model, n, model_type):
    return [generate_name(model, model_type) for _ in range(n)]

# 5. Initialization and Execution
hidden_size = 128
model_rnn = VanillaRNNScratch(vocab_size, 128).to(device)
model_blstm = BLSTMScratch(vocab_size, 64).to(device)
model_attn = AttentionRNNScratch(vocab_size, 128).to(device)

print("Training models (5 epochs each for speed)...")
train_model(model_rnn, names, epochs=5, model_type='rnn')
train_model(model_blstm, names, epochs=5, model_type='blstm')
train_model(model_attn, names, epochs=5, model_type='attention')

results = {}
for model, mtype, label in [(model_rnn, 'rnn', 'Vanilla RNN'), 
                            (model_blstm, 'blstm', 'BLSTM'), 
                            (model_attn, 'attention', 'Attention RNN')]:
    gen_names = generate_n_names(model, 500, mtype)
    
    # Novelty Rate: (names not in training) / total generated
    novel_count = sum(1 for n in gen_names if n not in train_set)
    novelty_rate = (novel_count / 500) * 100
    
    # Diversity: unique generated / total generated
    unique_names = set(gen_names)
    diversity = len(unique_names) / 500
    
    results[label] = (novelty_rate, diversity)
    
    # Save to txt
    filename = label.lower().replace(" ", "_") + "_metrics.txt"
    with open(filename, 'w') as f:
        f.write(f"Model: {label}\n")
        f.write(f"Novelty Rate: {novelty_rate:.2f}%\n")
        f.write(f"Diversity: {diversity:.4f}\n")
    print(f"Saved results for {label} to {filename}")

# Print summary
print("\nSummary:")
for model, (nov, div) in results.items():
    print(f"{model}: Novelty {nov:.2f}%, Diversity {div:.4f}")
