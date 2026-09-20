#!/usr/bin/env python3

import json
import heapq
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = Path("checkpoints/homegpt_v5_best.pt")
VOCAB_PATH = Path("tokenizer/vocab.json")
MERGES_PATH = Path("tokenizer/merges.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMPERATURE = 0.8
TOP_K = 50
MAX_NEW_TOKENS = 200


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


# ============================================================
# RoPE
# Exact same implementation used by train_homegpt_v5.py
# ============================================================

def build_rope_cache(seq_len, head_dim, device, theta=10000.0):
    assert head_dim % 2 == 0

    half = head_dim // 2

    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                half,
                device=device,
                dtype=torch.float32,
            )
            / half
        )
    )

    positions = torch.arange(
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    angles = torch.outer(positions, frequencies)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    T = x.shape[-2]
    D = x.shape[-1]

    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]

    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    rotated_first = x1 * cos - x2 * sin
    rotated_second = x1 * sin + x2 * cos

    return torch.cat(
        [rotated_first, rotated_second],
        dim=-1,
    )


# ============================================================
# SwiGLU
# ============================================================

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gated = F.silu(self.gate(x))
        return self.down(gated * self.up(x))


# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_hidden):
        super().__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.ffn = SwiGLU(dim, ffn_hidden)

    def forward(self, x, cos, sin):
        residual = x

        h = self.norm1(x)

        B, T, C = h.shape

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(B, T, C)

        x = residual + self.o_proj(attention)

        x = x + self.ffn(self.norm2(x))

        return x


# ============================================================
# HomeGPT V5
# Exact architecture from train_homegpt_v5.py
# ============================================================

class HomeGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd,
        n_head,
        n_layer,
        ffn_hidden,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.ffn_hidden = ffn_hidden

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd,
                    n_head,
                    ffn_hidden,
                )
                for _ in range(n_layer)
            ]
        )

        self.final_norm = RMSNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False,
        )

        # IMPORTANT:
        # Training uses tied token embedding / LM head.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"BLOCK_SIZE={self.block_size}"
            )

        x = self.token_embedding(idx)

        cos, sin = build_rope_cache(
            T,
            self.n_embd // self.n_head,
            x.device,
        )

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# EXACT tokenizer format used by tokenizer_final_gpu.py
#
# vocab.json:
#   list where index == token ID
#
# merges.json:
#   [
#       [[left_id, right_id], new_token_id],
#       ...
#   ]
#
# This mirrors FastBPEEncoder from tokenizer_final_gpu.py.
# ============================================================

class HomeGPTTokenizer:
    def __init__(self, vocab_path, merges_path):
        print("Loading tokenizer...")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        with open(merges_path, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        if not isinstance(self.vocab, list):
            raise ValueError(
                "Expected vocab.json to be a list."
            )

        self.id_to_token = {
            i: str(token)
            for i, token in enumerate(self.vocab)
        }

        self.token_to_id = {
            token: i
            for i, token in self.id_to_token.items()
        }

        self.special_tokens = [
            "<|pad|>",
            "<|unk|>",
            "<|bos|>",
            "<|eos|>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ]

        self.special_ids = {
            token: self.token_to_id[token]
            for token in self.special_tokens
            if token in self.token_to_id
        }

        self.unk_id = self.special_ids.get(
            "<|unk|>",
            1,
        )

        # Exact merge rank/result tables.
        self.rank = {}
        self.result = {}

        for rank, entry in enumerate(raw_merges):
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], list)
                or len(entry[0]) != 2
            ):
                raise ValueError(
                    f"Unexpected merge format at rank {rank}: {entry}"
                )

            left_id = int(entry[0][0])
            right_id = int(entry[0][1])
            new_id = int(entry[1])

            pair = (left_id, right_id)

            self.rank[pair] = rank
            self.result[pair] = new_id

        # Character IDs are exactly the IDs from the base vocabulary.
        # tokenizer_final_gpu.py assigns special IDs first, then characters.
        self.char_stoi = {}

        for token_id, token in self.id_to_token.items():
            if token_id < len(self.special_tokens):
                continue

            # Base characters are IDs before the first BPE-created token.
            # The tokenizer checkpoint tells us there are 6105 base chars.
            if token_id < len(self.special_tokens) + 6105:
                self.char_stoi[token] = token_id

        self.max_token_chars = max(
            (len(token) for token in self.vocab if token),
            default=1,
        )

        print(f"Vocabulary : {len(self.vocab):,}")
        print(f"Merges     : {len(self.rank):,}")
        print(f"Base chars : {len(self.char_stoi):,}")

    def encode_chunk(self, text):
        """
        Exact heap/linked-list BPE algorithm used by the tokenizer.
        """
        if not text:
            return []

        ids = [
            self.char_stoi.get(ch, self.unk_id)
            for ch in text
        ]

        n = len(ids)

        if n == 0:
            return []

        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1

        alive = [True] * n

        heap = []

        def push_pair(left):
            if left < 0 or not alive[left]:
                return

            right = nxt[left]

            if right < 0 or not alive[right]:
                return

            key = (ids[left], ids[right])
            rank = self.rank.get(key)

            if rank is not None:
                heapq.heappush(
                    heap,
                    (rank, left, right),
                )

        for i in range(n - 1):
            push_pair(i)

        while heap:
            rank, left, right = heapq.heappop(heap)

            if (
                left < 0
                or right < 0
                or not alive[left]
                or not alive[right]
                or nxt[left] != right
            ):
                continue

            key = (ids[left], ids[right])

            if self.rank.get(key) != rank:
                continue

            new_id = self.result[key]

            ids[left] = new_id
            alive[right] = False

            after = nxt[right]
            nxt[left] = after

            if after >= 0:
                prev[after] = left

            before = prev[left]

            push_pair(before)
            push_pair(left)

        output_ids = []

        i = 0

        while i >= 0 and i < n:
            if alive[i]:
                output_ids.append(ids[i])

            i = nxt[i]

        return output_ids

    def encode(self, text):
        """
        Encode ordinary text.

        Special tokens are preserved as atomic tokens.
        """
        if not text:
            return []

        output = []

        # We do not use regex here so that special tokens cannot be
        # accidentally altered by the BPE process.
        special_tokens = sorted(
            self.special_ids.keys(),
            key=len,
            reverse=True,
        )

        i = 0

        while i < len(text):
            matched = None

            for special in special_tokens:
                if text.startswith(special, i):
                    matched = special
                    break

            if matched is not None:
                output.append(self.special_ids[matched])
                i += len(matched)
                continue

            # Consume ordinary text until the next special token.
            start = i

            while i < len(text):
                if any(
                    text.startswith(s, i)
                    for s in special_tokens
                ):
                    break
                i += 1

            output.extend(
                self.encode_chunk(text[start:i])
            )

        return output

    def decode(self, ids):
        return "".join(
            self.id_to_token.get(
                int(idx),
                "<|unk|>",
            )
            for idx in ids
        )


# ============================================================
# Load model
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model = HomeGPT(
        vocab_size=int(checkpoint["vocab_size"]),
        block_size=int(checkpoint["block_size"]),
        n_embd=int(checkpoint["n_embd"]),
        n_head=int(checkpoint["n_head"]),
        n_layer=int(checkpoint["n_layer"]),
        ffn_hidden=int(checkpoint["ffn_hidden"]),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print()
    print("=" * 70)
    print("HOMEGPT V5")
    print("=" * 70)
    print(f"Checkpoint step : {checkpoint['step']}")
    print(f"Train loss      : {checkpoint.get('train_loss')}")
    print(f"Val loss        : {checkpoint.get('val_loss')}")
    print(f"Vocabulary      : {checkpoint['vocab_size']:,}")
    print(f"Context         : {checkpoint['block_size']}")
    print(f"Embedding       : {checkpoint['n_embd']}")
    print(f"Layers          : {checkpoint['n_layer']}")
    print(f"Heads           : {checkpoint['n_head']}")
    print(f"FFN hidden      : {checkpoint['ffn_hidden']}")
    print(f"Device          : {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            "GPU             :",
            torch.cuda.get_device_name(0),
        )

    print("=" * 70)

    return model


# ============================================================
# Generation
# Exact sampling logic from train_homegpt_v5.py
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
):
    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        token_ids = [
            tokenizer.special_ids.get(
                "<|bos|>",
                2,
            )
        ]

    idx = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]

        if DEVICE.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(idx_cond)
        else:
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        logits = logits / max(
            temperature,
            1e-5,
        )

        if top_k is not None:
            values, _ = torch.topk(
                logits,
                min(top_k, logits.shape[-1]),
            )

            logits[
                logits < values[:, [-1]]
            ] = float("-inf")

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        idx = torch.cat(
            [idx, next_token],
            dim=1,
        )

        # The training generate() does not explicitly stop on EOS,
        # so we intentionally do not add EOS stopping here.

    return tokenizer.decode(idx[0].tolist())


# ============================================================
# Tokenizer sanity test
# ============================================================

def tokenizer_test(tokenizer):
    test_texts = [
        "Artificial intelligence is",
        "Machine learning is",
        "The future of technology is",
    ]

    print()
    print("=" * 70)
    print("TOKENIZER SANITY CHECK")
    print("=" * 70)

    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        print()
        print(f"Text    : {text}")
        print(f"IDs     : {ids}")
        print(f"Decoded : {decoded}")
        print(f"PASS    : {decoded == text}")


# ============================================================
# Main
# ============================================================

def main():
    tokenizer = HomeGPTTokenizer(
        VOCAB_PATH,
        MERGES_PATH,
    )

    tokenizer_test(tokenizer)

    model = load_model()

    print()
    print("Type 'exit' to quit.")
    print()

    while True:
        try:
            prompt = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if prompt.strip().lower() in {"exit", "quit"}:
            break

        if not prompt.strip():
            continue

        generated = generate(
            model,
            tokenizer,
            prompt,
        )

        print()
        print("HomeGPT:")
        print(generated)
        print()


if __name__ == "__main__":
    main()
