# Audio-JEPA

A from-scratch implementation of **Joint Embedding Predictive Architecture (JEPA)** applied to audio, trained on LibriSpeech. The model learns to predict text-space embeddings directly from audio — without doing any transcription during pretraining — and then decodes those embeddings into text in a second, separate phase.

Built and trained on a single **RTX 3050 Laptop GPU**.

---

## Training Results

![Training Results](assets/training_results.png)

| Metric | Start | End |
|---|---|---|
| Phase 1 InfoNCE loss | 1.6498 | **0.0135** |
| Phase 1 alignment (cosine sim) | 0.0222 | **0.7690** |
| Phase 2 decoder loss (CE) | 10.7421 | **7.0039** |
| Phase 2 perplexity | 46,264 | **1,101** |

The alignment score rising from near-zero to 0.77 is the key result: the audio predictor learned to map speech to a point in embedding space that is close to the corresponding text embedding, without ever seeing text supervision during feature learning.

---

## What is JEPA?

**JEPA** (Joint Embedding Predictive Architecture), introduced by LeCun (2022) and realised as I-JEPA (Assran et al., 2023) for images, is a self-supervised learning framework built on a core idea:

> Instead of predicting raw inputs (pixels, audio samples), predict the **embedding** of a target in a learned representation space.

This is fundamentally different from:
- **Masked autoencoders (MAE):** predict raw tokens/pixels in input space
- **Contrastive methods (SimCLR, CLIP):** pull together views, push apart negatives
- **Generative methods (GPT-style):** predict the next raw token

JEPA avoids the "where to put the probability mass" problem of generative models and the collapse risk of pure contrastive methods.

**Audio-JEPA** extends this to the cross-modal setting: the prediction target is a **text embedding** rather than a masked audio segment. This forces the audio encoder to build representations that are semantically aligned with language.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │           PHASE 1 TRAINING                  │
                    │                                             │
  Raw Audio         │  ┌──────────────┐    ┌────────────────┐    │
  (16 kHz PCM) ─────┼─►│  X-Encoder   │───►│   Predictor    │    │
                    │  │ (HuBERT, ❄️)  │    │ (6× SWA block) │    │
                    │  └──────────────┘    └───────┬────────┘    │
                    │  768-dim frames              │ s_hat        │
                    │  @ 50 Hz                     │ (512-d)      │
                    │                              ▼              │
                    │                    ┌────────────────────┐   │
                    │                    │   InfoNCE Loss     │   │
                    │                    └────────────────────┘   │
                    │                              ▲              │
                    │  Text transcripts            │ s_y          │
                    │  "he said the..."  ─────────►│ (512-d)      │
                    │  ┌──────────────────────┐    │              │
                    │  │     Y-Encoder        │────┘              │
                    │  │  (MiniLM-L6, ❄️+proj)│                   │
                    │  └──────────────────────┘                   │
                    └─────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────┐
                    │           PHASE 2 TRAINING                  │
                    │                                             │
  Raw Audio ────────┼──► X-Encoder (❄️) ──► Predictor (❄️)        │
                    │                              │ s_hat        │
                    │                              ▼              │
                    │                    ┌────────────────────┐   │
                    │                    │    Y-Decoder       │   │
                    │                    │  (4× causal attn)  │   │
                    │                    └─────────┬──────────┘   │
                    │                              │              │
                    │                    Cross-entropy loss       │
                    │                    vs. ground-truth tokens  │
                    └─────────────────────────────────────────────┘
```

---



---

## Quick Start

```bash
git clone https://github.com/<you>/audio-jepa
cd audio-jepa
pip install -r requirements.txt

# Full training on LibriSpeech (streams ~2000 samples)
python train.py

# Quick sanity check with synthetic data (~30s)
python train.py --smoke_test --phase1_steps 50 --phase2_steps 20
```

---

## Mathematics

### 1. HuBERT Audio Encoder (X-Encoder)

HuBERT (Hidden-Unit BERT, Hsu et al., 2021) is a self-supervised speech model trained by predicting offline cluster assignments from masked audio segments. It consists of:

- **CNN feature extractor** — 7-layer 1D ConvNet that downsamples raw audio from 16 kHz to 50 Hz. Each output frame represents 20ms of audio.
- **Transformer encoder** — 12 transformer layers operating on the 768-dimensional frame sequence.

In Audio-JEPA, HuBERT is used **frozen** as a feature extractor only. Its weights are never updated. It converts raw PCM audio $a \in \mathbb{R}^{T_\text{audio}}$ to a frame sequence:

$$X = \text{HuBERT}(a) \in \mathbb{R}^{T_\text{frames} \times 768}$$

where $T_\text{frames} \approx T_\text{audio} / 320$ (the CNN downsampling factor).

**Why freeze HuBERT?** It already contains rich phonemic and acoustic representations learned from 960 hours of LibriSpeech. Training on it from scratch would require far more compute and data.

---

### 2. Sliding Window Attention (SWA)

Standard self-attention is $O(T^2)$ in the sequence length. For a 10-second clip at 50 Hz, that's $T = 500$ frames — manageable, but costly when stacked across 6 layers.

**Sliding window attention** restricts each position $i$ to attend only to positions $j$ with $|i - j| \leq W$:

$$\text{SWA}(Q, K, V)_i = \text{softmax}\!\left(\frac{Q_i \cdot K_{[i-W:i+W]}^\top}{\sqrt{d_h}} + M_i\right) V_{[i-W:i+W]}$$

where $M_i$ is a mask that sets all positions outside the window to $-\infty$.

**Complexity:** $O(T \cdot W \cdot d)$ instead of $O(T^2 \cdot d)$.

**Why it works for audio:** Speech has strong local temporal structure — phonemes and syllables span 20–200ms (1–10 frames). Long-range information propagates through stacked layers rather than a single-layer global attention.

**NaN handling:** Padding positions have all-$-\infty$ attention logits. After softmax: $\text{softmax}(-\infty, \ldots, -\infty) = \text{NaN}$. These are replaced with zeros via `nan_to_num`, producing a zero output vector for fully-padded positions. This is semantically correct — padded positions contribute nothing.

---

### 3. Predictor: Audio Sequence → Embedding

The predictor converts the variable-length HuBERT frame sequence $X \in \mathbb{R}^{T \times 768}$ into a single unit-norm vector $\hat{s} \in \mathbb{R}^{512}$.

**Architecture:**

$$x^{(0)} = X W_\text{in}, \quad W_\text{in} \in \mathbb{R}^{768 \times 512}$$

$$x^{(\ell)} = \text{PredictorBlock}(x^{(\ell-1)}), \quad \ell = 1, \ldots, 6$$

$$\bar{x} = \text{MaskedMeanPool}(x^{(6)})$$

$$\hat{s} = \frac{W_\text{out} \, \bar{x}}{\|W_\text{out} \, \bar{x}\|_2}$$

**Masked mean pooling** computes the average only over valid (non-padding) frames:

$$\bar{x} = \frac{\sum_{t=1}^{T} m_t \cdot x_t^{(6)}}{\sum_{t=1}^{T} m_t + \epsilon}$$

where $m_t \in \{0, 1\}$ is the validity mask.

**L2 normalisation** projects the output onto the unit hypersphere, which is required for the InfoNCE objective to be well-behaved (the text encoder output is also normalised, so their dot product is bounded in $[-1, 1]$).

**Each PredictorBlock uses SwiGLU:**

$$\text{FFN}(h) = W_\text{down}\!\left(\text{SiLU}(W_\text{gate} \, h) \odot W_\text{up} \, h\right)$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$. The multiplicative gating allows the network to selectively suppress activations, giving more representational power than a standard $\text{ReLU}$ or $\text{GELU}$ FFN.

---

### 4. Y-Encoder (Text Encoder)

Uses `all-MiniLM-L6-v2`, a sentence transformer pre-trained on 1B+ (sentence, paraphrase) pairs. It maps a text string to a 384-dimensional embedding.

For a transcript $y$, the text embedding is:

$$s_y = \frac{W_\text{proj} \cdot \text{MeanPool}(\text{MiniLM}(y))}{\|W_\text{proj} \cdot \text{MeanPool}(\text{MiniLM}(y))\|_2}$$

where $W_\text{proj} \in \mathbb{R}^{512 \times 384}$ is the only learnable part of the Y-Encoder. The MiniLM backbone is frozen.

**Mean pooling** is weighted by the attention mask to ignore padding tokens:

$$\text{MeanPool}(H, m) = \frac{\sum_t m_t h_t}{\sum_t m_t}$$

The goal of Phase 1 is to make $\hat{s} \approx s_y$ for aligned (audio, transcript) pairs — i.e., to pull the audio prediction into the same region of embedding space as the corresponding text.

---

### 5. InfoNCE Loss (Phase 1)

InfoNCE (van den Oord et al., 2018) is the noise-contrastive loss used in CLIP, SimCLR, and most modern multimodal contrastive systems.

Given a batch of $B$ audio-text pairs, the model produces:
- $\hat{S} \in \mathbb{R}^{B \times d}$ — audio predictions (rows are unit vectors)
- $S_Y \in \mathbb{R}^{B \times d}$ — text embeddings (rows are unit vectors)

The **similarity matrix** is:

$$L_{ij} = \frac{\hat{s}_i \cdot s_{y_j}}{\tau}$$

where $\tau > 0$ is the temperature. Since both are unit-normalised, $\hat{s}_i \cdot s_{y_j} \in [-1, 1]$.

The loss treats each row $i$ as a classification problem: predict that the positive pair is at column $i$:

$$\mathcal{L}_\text{audio \to text} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{L_{ii}}}{\sum_{j=1}^{B} e^{L_{ij}}}$$

$$\mathcal{L}_\text{text \to audio} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{L_{ii}}}{\sum_{j=1}^{B} e^{L_{ji}}}$$

$$\mathcal{L}_\text{InfoNCE} = \frac{\mathcal{L}_\text{audio \to text} + \mathcal{L}_\text{text \to audio}}{2}$$

**Lower bound on mutual information:** InfoNCE is a lower bound on $I(\hat{S}; S_Y)$. Minimising it maximises a lower bound on mutual information between the two modalities (van den Oord et al., 2018).

**Optimal temperature:** Too high ($\tau \to \infty$): all similarities are equal, loss is flat, no gradient. Too low ($\tau \to 0$): the loss concentrates on the hardest negative, ignoring easy ones — unstable. In practice, $\tau \approx 0.07$ is a good starting point (the CLIP value). The learnable temperature allows the model to adapt it.

**Temperature freezing:** The temperature is frozen at $\tau = 0.07$ for the first 200 steps. This prevents the logits from blowing up before the predictor has learned any useful structure — a common failure mode in contrastive training that produces NaN losses.

---

### 6. Alignment and Uniformity (Wang & Isola, 2020)

Beyond the InfoNCE loss value, two diagnostic metrics track representation quality:

**Alignment** measures how close positive pairs are on the unit sphere:

$$\mathcal{A} = \mathbb{E}_{(x, y) \sim \text{positive pairs}}\!\left[\hat{s}(x) \cdot s_y(y)\right]$$

Higher alignment means the audio embedding is closer to its paired text embedding. In training, this rose from $0.022$ (random) to $0.769$.

**Uniformity** measures how spread out embeddings are across the unit sphere:

$$\mathcal{U} = \mathbb{E}_{x \neq x'}\!\left[\|\hat{s}(x) - \hat{s}(x')\|^2\right]$$

High uniformity means the encoder is using the embedding space efficiently (not collapsing all embeddings to a single point). The uniformity-alignment trade-off is the central tension in contrastive learning.

---

### 7. Y-Decoder: Autoregressive Text Generation (Phase 2)

The decoder is a small causal transformer conditioned on the predictor output $\hat{s} \in \mathbb{R}^{512}$.

**Conditioning:** $\hat{s}$ is projected to the decoder dimension $d = 256$ and used as the single memory token:

$$m = W_\text{ctx} \, \hat{s} \in \mathbb{R}^{d}$$

The decoder then cross-attends to this single memory vector.

**Token embeddings:**

$$e_t = \text{TokEmb}(y_t) + \text{PosEmb}(t)$$

**Causal transformer decoder (4 layers, 4 heads):**

At each layer $\ell$, with self-attention masked causally and cross-attention over $m$:

$$h_t^{(\ell)} = \text{CrossAttn}\!\left(\text{SelfAttn}(h_t^{(\ell-1)}),\; m\right)$$

**Output logits:**

$$p(y_{t+1} \mid y_{1:t}, \hat{s}) = \text{softmax}(W_\text{head} \cdot \text{LN}(h_t^{(L)}))$$

**Phase 2 loss (teacher-forcing):**

$$\mathcal{L}_\text{CE} = -\sum_{t=1}^{T-1} \log p(y_{t+1} \mid y_{1:t}, \hat{s})$$

The targets are the ground-truth token ids shifted by one position. Padding token positions are ignored (via `ignore_index` in `CrossEntropyLoss`).

**Phase 2 perplexity:** $\text{PPL} = e^{\mathcal{L}_\text{CE}}$. Starting at $\sim$46,000 and falling to $\sim$1,100 over 500 steps, this confirms the decoder is learning to associate context vectors with coherent token sequences.

**Why high perplexity remains:** The decoder is conditioned on a single 512-d vector — a severe compression of an entire utterance. At 500 steps with 2000 training samples, the decoder hasn't generalised. Longer training, more data, and beam search would improve output quality.

---

### 8. Mixed Precision (AMP)

Both phases use PyTorch Automatic Mixed Precision (`torch.amp.autocast`):

- **Forward pass:** FP16 matmuls on GPU (2× faster, 2× less memory)
- **Loss, backward:** Kept in FP32 for numerical stability
- **GradScaler:** Scales loss up before backward to prevent FP16 underflow of small gradients; unscales before the optimizer step

This gave approximately 2× throughput improvement on the RTX 3050, processing ~8,000 tokens/batch per second.

---

### 9. LR Schedule

$$\eta(t) = \eta_\text{max} \cdot \begin{cases}
t / T_\text{warmup} & t < T_\text{warmup} \\
0.1 + 0.9 \cdot \frac{1 + \cos\!\left(\pi \cdot \frac{t - T_\text{warmup}}{T_\text{total} - T_\text{warmup}}\right)}{2} & t \geq T_\text{warmup}
\end{cases}$$

- Phase 1: $\eta_\text{max} = 10^{-4}$, warmup 100 steps, total 1000 steps
- Phase 2: $\eta_\text{max} = 10^{-4}$, warmup 50 steps, total 500 steps
- Minimum ratio: $0.1$ (LR never drops below $10^{-5}$)

---

### 10. Gradient Accumulation

With `BATCH_SIZE=8` and `GRAD_ACCUM=4`, the effective batch size is $8 \times 4 = 32$. Gradients are accumulated over 4 mini-batches before each optimizer step, simulating a larger batch without the GPU memory cost.

---

### 11. Data Pipeline

LibriSpeech articles are decoded with `soundfile` (bypassing the default `torchcodec` which requires FFmpeg on Windows). Each audio sample is:

1. **Loaded and decoded** from bytes or file path
2. **Normalised:** $(a - \mu) / (\sigma + \epsilon)$, then clipped to $[-3\sigma, 3\sigma]$ — this maps speech amplitude to a consistent range regardless of recording volume
3. **Truncated** to a maximum of 20 seconds ($320,000$ samples)
4. **Padded** in the collate function to the longest sample in each batch

**Attention mask:** A binary mask of shape $(B, T)$ tracks which positions are real audio vs. zero padding. This mask is passed through HuBERT and the predictor's mean-pooling.

---

## Model Configuration

| Component | Model | Params | Status |
|---|---|---|---|
| X-Encoder | HuBERT-base-LS960 | 94.4M | Frozen |
| Predictor | 6× SWA blocks, d=512 | 17.4M | Trained (Phase 1) |
| Y-Encoder | MiniLM-L6-v2 + proj | 22.7M | proj only (Phase 1) |
| Y-Decoder | 4× causal blocks, d=256 | ~5.5M | Trained (Phase 2) |
| **Total** | | **163.9M** | |
| **Trainable** | | **69.5M** | |

---

## Training Details

| Setting | Value |
|---|---|
| Hardware | NVIDIA RTX 3050 Laptop GPU (4.3 GB VRAM) |
| Dataset | LibriSpeech train-clean-100 (2000 samples) |
| Batch size | 8 × 4 (grad accum) = 32 effective |
| Phase 1 steps | 1,000 |
| Phase 2 steps | 500 |
| Optimizer | Adam ($\beta_1=0.9$, $\beta_2=0.999$) |
| Peak LR | $10^{-4}$ |
| Gradient clip | 1.0 |
| Embed dim | 512 |
| AMP | FP16 (GradScaler) |
| GPU memory | ~1.1 GB (Phase 1), ~1.4 GB (Phase 2) |

---

## References

1. **LeCun, Y.** (2022). A Path Towards Autonomous Machine Intelligence. OpenReview.
2. **Assran et al.** (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). CVPR 2023.
3. **Hsu et al.** (2021). HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM TASLP.
4. **van den Oord et al.** (2018). Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748.
5. **Radford et al.** (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP). ICML 2021.
6. **Wang & Isola** (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. ICML 2020.
7. **Shazeer, N.** (2020). GLU Variants Improve Transformer. arXiv:2002.05202. (SwiGLU)
8. **Reimers & Gurevych** (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.
9. **Panayotov et al.** (2015). LibriSpeech: An ASR Corpus Based on Public Domain Audio Books. ICASSP 2015.

---

## License

MIT
