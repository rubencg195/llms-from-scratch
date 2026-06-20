---
jupytext:
  text_representation:
    extension: .md
    format_name: light
    format_version: '1.5'
kernelspec:
  display_name: Python 3 (LLMs)
  language: python
  name: python3
---

# Section 7.2: Discretizing Sound — Converting Audio Waves into Integer Tokens

**Goal:** Load a waveform, simulate codec frame tokens, and treat them like text IDs for autoregressive modeling.

## What You Need to Know First

We turn sound into the same kind of integer tokens you already train on. Background you need is small:

- **From the text phases:** tokens are just integers from a fixed vocabulary, and the model predicts the next one ("autoregressive" = predict the next token from the previous ones).
- **BPE as compression** — recall that tokenization turns a long raw signal (characters) into a shorter sequence of symbols. We do the same thing to audio.
- **A waveform** — digital sound is literally a long list of numbers (air-pressure measurements), nothing more.
- **High-school algebra** — sine waves and averages are all the math used; library calls handle the rest.

No prior signal-processing or audio experience is assumed.

## Sound is Just a Really Fast Sequence of Numbers

At its core, digital audio is simple: a microphone measures air pressure 24,000 times per
second (24 kHz), producing a sequence of floating-point numbers. One second of speech =
24,000 numbers. One minute = 1,440,000 numbers.

We cannot feed this directly to an LLM — the sequence length would be absurd. Instead, we
**compress** the waveform into discrete tokens at a much lower rate (typically 12.5–50 Hz),
giving us 12–50 tokens per second. This is the same idea as BPE for text: compressing a raw
sequence into a shorter sequence of discrete symbols.

The pipeline: `Waveform → Mel Spectrogram → Vector Quantize → Integer Tokens`

---

## Generating a Synthetic Waveform

```python
import torch
import numpy as np

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

sample_rate = 24000
duration_s = 1.0
t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
# A4 note (440 Hz) with harmonics for richer sound
wave = (0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 880 * t) +
        0.1 * np.sin(2 * np.pi * 1320 * t))

if HAS_SF:
    sf.write("tone.wav", wave.astype(np.float32), sample_rate)
    print("Wrote tone.wav")

print(f"Waveform: {wave.shape[0]} samples at {sample_rate} Hz = {duration_s}s")
print(f"Raw values per second: {sample_rate:,} (too many for an LLM!)")
```

---

## Mel Spectrograms: Why Frequency-Domain is Better

Time-domain waveforms contain massive redundancy. A mel spectrogram captures the *perceptually
relevant* frequency content at each time step, matching how the human ear processes sound.
This reduces dimensionality dramatically while preserving what matters for speech.

```python
def simple_mel_spectrogram(wave, sr, n_fft=1024, hop=512, n_mels=80):
    """
    Simplified mel spectrogram computation.
    In production, use librosa or torchaudio.
    """
    # STFT
    n_frames = 1 + (len(wave) - n_fft) // hop
    frames = np.zeros((n_frames, n_fft))
    window = np.hanning(n_fft)
    for i in range(n_frames):
        start = i * hop
        frames[i] = wave[start:start + n_fft] * window

    # FFT
    spectrum = np.abs(np.fft.rfft(frames, axis=1))  # (n_frames, n_fft//2 + 1)

    # Mel filter bank (simplified linear spacing)
    n_freq = spectrum.shape[1]
    mel_filters = np.zeros((n_mels, n_freq))
    freq_bins = np.linspace(0, n_freq - 1, n_mels + 2).astype(int)
    for m in range(n_mels):
        start, center, end = freq_bins[m], freq_bins[m+1], freq_bins[m+2]
        for k in range(start, center):
            mel_filters[m, k] = (k - start) / max(center - start, 1)
        for k in range(center, end):
            mel_filters[m, k] = (end - k) / max(end - center, 1)

    mel_spec = np.log1p(spectrum @ mel_filters.T)  # (n_frames, n_mels)
    return mel_spec

mel = simple_mel_spectrogram(wave, sample_rate)
print(f"Mel spectrogram shape: {mel.shape}")
print(f"Compression: {sample_rate} samples/s -> {mel.shape[0]} frames/s")
print(f"Each frame: {mel.shape[1]} mel bins (vs {sample_rate} raw samples)")
```

---

## Vector Quantization with Codebook Learning (Toy k-means)

The key idea: learn a **codebook** of representative audio patterns, then replace each
frame with the index of its nearest codebook entry. This converts continuous mel frames
into discrete integer tokens.

```python
def train_codebook_kmeans(data, codebook_size, n_iter=20):
    """
    Train a VQ codebook using k-means on frame data.
    data: (N, D) — N frames, D dimensions
    Returns: codebook (codebook_size, D)
    """
    N, D = data.shape
    # Initialize with random data points
    indices = np.random.choice(N, codebook_size, replace=False)
    codebook = data[indices].copy()

    for iteration in range(n_iter):
        # Assign each frame to nearest codebook entry
        dists = np.sum((data[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
        assignments = np.argmin(dists, axis=1)

        # Update codebook entries to mean of assigned frames
        for k in range(codebook_size):
            mask = assignments == k
            if mask.sum() > 0:
                codebook[k] = data[mask].mean(axis=0)

    return codebook, assignments

np.random.seed(42)
codebook_size = 256
codebook_np, assignments = train_codebook_kmeans(mel, codebook_size, n_iter=30)
print(f"Codebook shape: {codebook_np.shape}")
print(f"Token assignments: {assignments[:20]}")
print(f"Unique tokens used: {len(np.unique(assignments))} / {codebook_size}")
```

---

## Residual Vector Quantization (RVQ)

A single codebook may not capture all the detail. **RVQ** applies multiple quantization
levels: the first codebook captures coarse structure, the second captures the residual
error, the third captures the residual of the residual, etc. (A "residual" is simply the
leftover difference between the original and the current approximation.) This is how EnCodec and
Mimi — neural audio *codecs*, i.e. learned compressors that turn sound into tokens and back —
achieve high-quality audio with few tokens.

```python
def residual_vq(data, codebook_size, n_levels=3, n_iter=20):
    """
    Multi-level residual vector quantization.
    Returns: list of (codebook, token_ids) for each level.
    """
    residual = data.copy()
    levels = []

    for level in range(n_levels):
        codebook, assignments = train_codebook_kmeans(residual, codebook_size, n_iter)
        # Compute reconstruction at this level
        reconstruction = codebook[assignments]
        # Next level operates on the residual
        residual = residual - reconstruction
        levels.append((codebook, assignments))
        residual_energy = np.mean(residual ** 2)
        print(f"Level {level}: residual energy = {residual_energy:.6f}")

    return levels

print("Training 3-level RVQ:")
levels = residual_vq(mel, codebook_size=64, n_levels=3, n_iter=20)

# Total reconstruction
total_recon = np.zeros_like(mel)
for codebook, assignments in levels:
    total_recon += codebook[assignments]

recon_error = np.mean((mel - total_recon) ** 2)
single_error = np.mean((mel - levels[0][0][levels[0][1]]) ** 2)
print(f"\nSingle-level reconstruction error: {single_error:.6f}")
print(f"3-level RVQ reconstruction error:  {recon_error:.6f}")
print(f"Improvement: {single_error / max(recon_error, 1e-8):.1f}x better")
```

---

## Frame-Level Tokenization (Production-Style)

```python
frame_hop = 1920  # ~12.5 Hz at 24kHz -> 12.5 tokens per second

def frame_signal(wave, hop):
    frames = []
    for i in range(0, len(wave) - hop, hop):
        frames.append(wave[i : i + hop])
    return np.stack(frames)

frames = frame_signal(wave, frame_hop)
print(f"Num codec frames: {frames.shape[0]}")
print(f"Token rate: {frames.shape[0] / duration_s:.1f} tokens/sec")

# Quantize frames with a larger codebook
torch.manual_seed(0)
codebook_size_prod = 1024
codebook = torch.randn(codebook_size_prod, frames.shape[1])
frame_t = torch.from_numpy(frames).float()
dists = (frame_t.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
tokens = dists.argmin(dim=-1)
print(f"Audio token IDs: {tokens.tolist()}")
print(f"These are now ready for autoregressive modeling (just like text tokens)!")
```

---

## Visualization: Waveform, Spectrogram, and Token Indices

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Waveform
axes[0].plot(t[:2000], wave[:2000], linewidth=0.5, color='steelblue')
axes[0].set_title('Raw Waveform (first 2000 samples)')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')

# Mel spectrogram
axes[1].imshow(mel.T, aspect='auto', origin='lower', cmap='magma')
axes[1].set_title('Mel Spectrogram')
axes[1].set_xlabel('Frame Index')
axes[1].set_ylabel('Mel Bin')

# Token indices
token_indices = levels[0][1]  # first-level VQ assignments
axes[2].step(range(len(token_indices)), token_indices, where='mid', color='darkgreen')
axes[2].set_title('Quantized Token Indices (Level 0)')
axes[2].set_xlabel('Frame Index')
axes[2].set_ylabel('Codebook Index')

plt.tight_layout()
plt.savefig('audio_tokenization.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved audio_tokenization.png")
```

---

## Autoregressive Target

Predict `tokens[t+1]` from `tokens[:t]` plus **system tag** (Section 7.3). This is
identical to next-token prediction for text — the only difference is the vocabulary
represents audio codebook entries rather than BPE subwords.

---

## Exercise: Compare VQ Reconstruction Error Across Codebook Sizes

```python
print("Codebook Size vs Reconstruction Error:")
print(f"{'Codebook Size':<15} {'MSE':<12} {'Utilization'}")
print("-" * 45)

for cb_size in [64, 256, 1024, 4096]:
    np.random.seed(42)
    if cb_size <= mel.shape[0]:
        cb, assigns = train_codebook_kmeans(mel, cb_size, n_iter=30)
        recon = cb[assigns]
        mse = np.mean((mel - recon) ** 2)
        utilization = len(np.unique(assigns)) / cb_size * 100
        print(f"{cb_size:<15} {mse:<12.6f} {utilization:.1f}%")
    else:
        # Not enough frames for this codebook size, use smaller init
        cb, assigns = train_codebook_kmeans(mel, min(cb_size, mel.shape[0]), n_iter=30)
        recon = cb[assigns]
        mse = np.mean((mel - recon) ** 2)
        utilization = len(np.unique(assigns)) / min(cb_size, mel.shape[0]) * 100
        print(f"{cb_size:<15} {mse:<12.6f} {utilization:.1f}% (limited by data)")

print("\nTakeaway: larger codebooks reduce error but risk 'codebook collapse'")
print("(many entries go unused). RVQ is more parameter-efficient than huge single codebooks.")
```

---

## Where This Leads Next

Now that sound is just a stream of integer tokens, Section 7.3 asks the model to predict *two*
things at once: the next audio token (what to say) and a control tag (the conversation state
from Section 7.1). That dual-stream loss is what wires audio generation to the full-duplex logic.

## Key Takeaway

Audio tokenization bridges the gap between continuous sound and discrete language modeling.
By converting 24,000 samples/sec into ~12 tokens/sec via mel spectrograms and vector
quantization, we make audio amenable to the same autoregressive training used for text.
Residual VQ (multiple codebook levels) achieves high reconstruction quality while keeping
the per-frame token count low. In production, swap our toy VQ for pretrained **Mimi** or
**EnCodec** encoders — the principle is identical.

## Further Reading (Optional)

**Optional — you do NOT need these to continue. They are for curious students who want the original sources.**

- van den Oord, Vinyals, & Kavukcuoglu (2017). *Neural Discrete Representation Learning (VQ-VAE)*. NeurIPS.
- Défossez et al. (2022). *High Fidelity Neural Audio Compression (EnCodec)*. arXiv:2210.13438.
- Zeghidour et al. (2021). *SoundStream: An End-to-End Neural Audio Codec*. IEEE/ACM TASLP.
