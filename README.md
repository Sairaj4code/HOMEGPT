# HomeGPT

A small GPT-style language model built **from scratch in PyTorch**.

HomeGPT is an experimental project focused on understanding how modern language models work by implementing the major components myself rather than using a pretrained Hugging Face model.

The project includes a custom GPU-accelerated BPE tokenizer, a decoder-only Transformer architecture, custom training pipeline, checkpointing, and text generation.

> **Status:** Experimental / V5  
> HomeGPT currently generates text but is not yet instruction-tuned or optimized to behave like a conversational assistant.

---

## Features

- Custom BPE tokenizer built from scratch
- GPU-accelerated BPE vocabulary learning
- 8,192-token vocabulary
- ~6,105 base character tokens
- 2,000+ learned BPE merges
- Decoder-only Transformer architecture
- RMSNorm
- Rotary Positional Embeddings (RoPE)
- SwiGLU feed-forward network
- Causal self-attention
- PyTorch scaled dot-product attention
- Weight tying between token embeddings and language-model head
- Mixed-precision training with BF16/FP16
- Gradient accumulation
- AdamW optimizer
- Learning-rate warmup and decay
- Gradient clipping
- Validation loss evaluation
- Best-checkpoint saving
- Custom inference and text generation

---

## Architecture

HomeGPT V5 uses a small decoder-only Transformer architecture.

| Component | Configuration |
|---|---:|
| Vocabulary | 8,192 |
| Context length | 256 tokens |
| Embedding dimension | 320 |
| Transformer layers | 6 |
| Attention heads | 8 |
| FFN hidden dimension | 853 |
| Parameters | ~10 million |
| Normalization | RMSNorm |
| Positional encoding | RoPE |
| Activation | SwiGLU |
| Attention | Causal SDPA |
| Dropout | 0 |

The model uses **pre-normalization** and **tied input/output embeddings**.

---

## Model Architecture

The basic flow is:

```text
Input Text
    │
    ▼
Custom BPE Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Token Embeddings
    │
    ▼
┌──────────────────────────┐
│ Transformer Block × 6    │
│                          │
│ RMSNorm                  │
│   ↓                      │
│ Causal Self Attention    │
│ + RoPE                   │
│   ↓                      │
│ Residual Connection      │
│                          │
│ RMSNorm                  │
│   ↓                      │
│ SwiGLU Feed Forward      │
│   ↓                      │
│ Residual Connection      │
└──────────────────────────┘
    │
    ▼
Language Model Head
    │
    ▼
Next Token Prediction
