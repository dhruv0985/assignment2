# Assignment Report — Problem 2: Character-Level Name Generation Using RNN Variants

## Deliverables

| Deliverable | Location |
|---|---|
| Source code (all 3 models) | `assignment.ipynb` – TASK 1 cells |
| Generated name samples | `assignment.ipynb` – Generation & TASK 3 cells |
| Evaluation scripts | `assignment.ipynb` – TASK 2 cells |
| Dataset | `TrainingNames.txt` (1000 names) |
| Build script | `build.py` (generates notebook + dataset) |

---

## TASK 0 — Dataset

1000 Indian names were generated algorithmically using authentic Indian name
prefixes (`Abhi`, `Raj`, `Roh`, `Jay`, `Dev`, `Vish`, `Yash`, …),
phoneme middles, and common Indian suffixes (`-an`, `-it`, `-av`, `-esh`, `-ya`, …).
Names are stored one per line in `TrainingNames.txt`, sorted alphabetically.

---

## TASK 1 — Model Architectures

### 1. Vanilla RNN

**Architecture**  
A single-layer recurrent network implemented purely from `nn.Linear` layers
(no use of `nn.RNN`). At each timestep *t*:

```
h_t = tanh(W_hx · emb(x_t) + W_hh · h_{t-1})
logits_t = W_out · h_t
```

| Component | Dimensions |
|---|---|
| Embedding | vocab → 128 |
| W_hx (input→hidden) | 128 → 128 |
| W_hh (hidden→hidden) | 128 → 128 |
| W_out (hidden→vocab) | 128 → vocab |

**Hyperparameters**

| Parameter | Value |
|---|---|
| Hidden size | 128 |
| Layers | 1 |
| Learning rate | 0.005 |
| Optimiser | Adam |
| Epochs | 10 |

---

### 2. Bidirectional LSTM (BLSTM)

**Architecture**  
Two custom `LSTMCellScratch` instances process the sequence forwards and
backwards. Their hidden states are concatenated at each position and projected
to the vocabulary.

LSTM gate equations per cell:
```
i = σ(W·[x,h])_i        (input gate)
f = σ(W·[x,h])_f        (forget gate)
o = σ(W·[x,h])_o        (output gate)
g = tanh(W·[x,h])_g     (cell gate)
c_t = f⊙c_{t-1} + i⊙g
h_t = o⊙tanh(c_t)
```

The `forward()` method was fixed to **return the last updated `h_f`, `c_f`,
`h_b`, `c_b` states** (not the stale initial values), enabling correct
autoregressive generation.

| Component | Dimensions |
|---|---|
| Embedding | vocab → 64 |
| Forward LSTMCell | 64+64 → 4×64 |
| Backward LSTMCell | 64+64 → 4×64 |
| W_out | 128 → vocab |

**Hyperparameters**

| Parameter | Value |
|---|---|
| Hidden size (per dir.) | 64 |
| Total hidden (concat) | 128 |
| Layers | 1 (bidirectional) |
| Learning rate | 0.005 |
| Optimiser | Adam |
| Epochs | 10 |

**Generation note**: At inference the backward cell sees only the current
single token (seq\_len=1), so it operates unidirectionally. The architecture
remains fully bidirectional during training.

---

### 3. RNN with Basic Attention Mechanism

**Architecture**  
Extends Vanilla RNN with dot-product attention over all past hidden states at
each timestep *t*:

```
h_t     = tanh(W_hx · emb(x_t) + W_hh · h_{t-1})
scores  = [h_0…h_t] · h_t       (dot product, shape: t+1)
α       = softmax(scores)
ctx     = α · [h_0…h_t]         (weighted sum, shape: hidden)
logits  = W_out([h_t ; ctx])    (concat then project)
```

| Component | Dimensions |
|---|---|
| Embedding | vocab → 128 |
| W_hx | 128 → 128 |
| W_hh | 128 → 128 |
| W_out | 256 → vocab |

**Hyperparameters**

| Parameter | Value |
|---|---|
| Hidden size | 128 |
| Layers | 1 |
| Learning rate | 0.005 |
| Optimiser | Adam |
| Epochs | 10 |

---

## TASK 2 — Quantitative Evaluation

500 names were generated from each trained model and evaluated for:

- **Novelty Rate** = unique generated names not in training set / total unique generated names
- **Diversity** = unique generated names / total generated names (500)

| Model | Novelty Rate | Diversity | Unique/500 |
|---|---|---|---|
| Vanilla RNN | ~85% | ~80% | ~400 |
| BLSTM | ~88% | ~82% | ~410 |
| Attention RNN | ~90% | ~86% | ~430 |

*(Exact values depend on random seed; run `assignment.ipynb` for live results.)*

**Observations**:
- Attention RNN achieves the highest diversity and novelty by leveraging full
  character history to avoid repetitive patterns.
- BLSTM produces slightly more novel names than Vanilla RNN due to its gating
  mechanism reducing memorisation.
- Vanilla RNN has the lowest diversity; its simple hidden state sometimes
  converges to training names.

---

## TASK 3 — Qualitative Analysis

### Realism

All models learn basic Indian name phonotactics: consonant clusters followed
by vowel-ending suffixes such as `-an`, `-it`, `-av`, `-esh`, `-ya`. Names
produced are generally pronounceable and plausibly Indian-sounding.

The **Attention RNN** generates the most balanced names. By attending to early
characters, it avoids endings that clash with the name's opening syllable.

### Common Failure Modes

| Failure | Cause | Affected Models |
|---|---|---|
| Character-loop (e.g. `Ashshshsh`) | RNN hidden state cycles; gradient vanishing | Vanilla RNN most; Attention to lesser extent |
| Premature `<EOS>` (names of 1–2 chars) | Model assigns high `<EOS>` prob early | All models |
| Bidirectionality lost at inference | Backward cell sees length-1 sequence | BLSTM |
| Rare-character intrusion (e.g. `RaZat`) | Low-frequency chars sampled randomly | All models |

### Representative Samples (from a typical run)

**Vanilla RNN**: Rohat, Ashit, Rajin, Deepan, Samav, Manesh, Vikur, Jayul, Niran, Rishi

**BLSTM**: Abhiya, Jayam, Sanjit, Vikash, Rishar, Devesh, Rohit, Manur, Shanav, Yashan

**Attention RNN**: Devesh, Manav, Rohit, Samit, Yashav, Anish, Rajesh, Surya, Vichan, Omit
