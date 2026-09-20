"""
HomeGPT V5 - From-scratch Transformer training

This is the first serious HomeGPT language-model trainer.

Architecture
------------
- decoder-only causal Transformer
- RMSNorm
- RoPE positional encoding
- PyTorch scaled_dot_product_attention (CUDA optimized)
- SwiGLU feed-forward blocks
- pre-norm residual blocks
- tied token embedding / LM head
- mixed precision on CUDA
- gradient accumulation
- cosine learning-rate schedule with warmup
- gradient clipping
- validation loss
- best-checkpoint saving
- resumable checkpoints

Designed for a 4 GB RTX 3050.

IMPORTANT
---------
This script expects the final tokenizer to have produced:

    tokenizer/vocab.json
    tokenizer/merges.json
    tokenizer/bpe_checkpoint.json
    data/train_tokens.pt
    data/val_tokens.pt

The token tensors must contain IDs in [0, len(vocab)-1].

Why save token tensors?
-----------------------
Re-running a Python BPE encoder over a 13M-character corpus for every
training launch would waste time. Tokenization should be a one-time
data-preparation step; model training should consume compact integer IDs.
"""

import argparse
import copy
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")
CHECKPOINT_DIR = Path("checkpoints")

TRAIN_PATH = DATA_DIR / "train_tokens.pt"
VAL_PATH = DATA_DIR / "val_tokens.pt"

VOCAB_PATH = TOKENIZER_DIR / "vocab.json"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------- Model ----------------

BLOCK_SIZE = 256
N_EMBD = 320
N_HEAD = 8
N_LAYER = 6

# SwiGLU hidden dimension.
# 2/3 * 4 * d is close to the parameter-efficient SwiGLU convention.
FFN_MULTIPLIER = 4
FFN_HIDDEN = int(
    (2 / 3) * FFN_MULTIPLIER * N_EMBD
)

DROPOUT = 0.0

# ---------------- Training ----------------

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16

MAX_STEPS = 20000

# Stop when validation has not improved for this many evals.
EARLY_STOP_PATIENCE = 8

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

WARMUP_STEPS = 500

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 500
EVAL_ITERS = 50

SAVE_INTERVAL = 500

# ---------------- System ----------------

SEED = 1337

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# TF32 can significantly improve matmul throughput on
# supported NVIDIA GPUs without changing model architecture.
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# Use BF16 when supported; otherwise FP16.
if DEVICE.type == "cuda":
    if torch.cuda.is_bf16_supported():
        AMP_DTYPE = torch.bfloat16
    else:
        AMP_DTYPE = torch.float16
else:
    AMP_DTYPE = torch.float32


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(
            dim=-1,
            keepdim=True
        )

        x = x * torch.rsqrt(
            rms + self.eps
        )

        return x * self.weight


# ============================================================
# Rotary positional embeddings
# ============================================================

def build_rope_cache(
    seq_len,
    head_dim,
    device,
    theta=10000.0,
):
    """
    Precompute cos/sin tables for RoPE.

    RoPE gives attention access to token positions without a
    learned position embedding table.
    """

    assert head_dim % 2 == 0

    half = head_dim // 2

    frequencies = 1.0 / (
        theta ** (
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

    angles = torch.outer(
        positions,
        frequencies,
    )

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    x shape:
        B, heads, T, head_dim

    Rotate pairs of dimensions.
    """

    T = x.shape[-2]
    D = x.shape[-1]

    x1 = x[..., : D // 2]
    x2 = x[..., D // 2 :]

    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    rotated_first = (
        x1 * cos
        - x2 * sin
    )

    rotated_second = (
        x1 * sin
        + x2 * cos
    )

    return torch.cat(
        [
            rotated_first,
            rotated_second,
        ],
        dim=-1,
    )


# ============================================================
# SwiGLU
# ============================================================

class SwiGLU(nn.Module):

    def __init__(
        self,
        dim,
        hidden_dim,
    ):
        super().__init__()

        self.gate = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.up = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.down = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x):

        gated = F.silu(
            self.gate(x)
        )

        return self.down(
            gated * self.up(x)
        )


# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        ffn_hidden,
    ):
        super().__init__()

        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

        self.q_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.ffn = SwiGLU(
            dim,
            ffn_hidden,
        )

    def forward(
        self,
        x,
        cos,
        sin,
    ):

        residual = x

        h = self.norm1(x)

        B, T, C = h.shape

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        q = apply_rope(
            q,
            cos,
            sin,
        )

        k = apply_rope(
            k,
            cos,
            sin,
        )

        # PyTorch selects an optimized CUDA attention kernel
        # when available. is_causal=True supplies the causal mask.
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        attention = attention.transpose(
            1,
            2,
        ).contiguous()

        attention = attention.view(
            B,
            T,
            C,
        )

        x = residual + self.o_proj(
            attention
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# HomeGPT
# ============================================================

class HomeGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            N_EMBD,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    N_EMBD,
                    N_HEAD,
                    FFN_HIDDEN,
                )
                for _ in range(N_LAYER)
            ]
        )

        self.final_norm = RMSNorm(
            N_EMBD
        )

        # Precompute RoPE once instead of rebuilding it every forward pass.
        rope_cos, rope_sin = build_rope_cache(
            BLOCK_SIZE,
            N_EMBD // N_HEAD,
            torch.device("cpu"),
        )
        self.register_buffer(
            "rope_cos",
            rope_cos,
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            rope_sin,
            persistent=False,
        )

        self.lm_head = nn.Linear(
            N_EMBD,
            vocab_size,
            bias=False,
        )

        # Weight tying:
        # input token embeddings and output token projection
        # share the same parameter matrix.
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(
            self._init_weights
        )

    @staticmethod
    def _init_weights(module):

        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        idx,
        targets=None,
    ):

        B, T = idx.shape

        if T > BLOCK_SIZE:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"BLOCK_SIZE={BLOCK_SIZE}"
            )

        x = self.token_embedding(idx)

        cos = self.rope_cos[:T].to(
            device=x.device,
            dtype=x.dtype,
        )
        sin = self.rope_sin[:T].to(
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks:
            x = block(
                x,
                cos,
                sin,
            )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.vocab_size,
                ),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=50,
    ):

        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[
                :,
                -BLOCK_SIZE:,
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[:, -1, :]

            logits = (
                logits
                / max(temperature, 1e-5)
            )

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.shape[-1],
                    ),
                )

                logits[
                    logits
                    < values[:, [-1]]
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
                [
                    idx,
                    next_token,
                ],
                dim=1,
            )

        return idx


# ============================================================
# Dataset
# ============================================================

def load_tokens(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing token file: {path}\n"
            "Run the tokenizer/data-preparation stage first."
        )

    # These files are tensors produced by our own tokenizer. Using
    # weights_only=True avoids arbitrary pickle deserialization.
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(data, torch.Tensor):
        raise TypeError(
            f"{path} did not contain a tensor."
        )

    if data.ndim != 1:
        raise ValueError(
            f"{path} must be a 1-D token tensor; got shape {tuple(data.shape)}"
        )

    return data


train_data = load_tokens(
    TRAIN_PATH
)

val_data = load_tokens(
    VAL_PATH
)


def validate_dataset(data, name, vocab_size):
    if data.numel() == 0:
        raise ValueError(f"{name} dataset is empty.")

    min_id = int(data.min().item())
    max_id = int(data.max().item())

    if min_id < 0 or max_id >= vocab_size:
        raise ValueError(
            f"{name} token IDs out of range: "
            f"{min_id}..{max_id}, vocab={vocab_size}"
        )

    print(
        f"{name:>12} tokens: {data.numel():,} | "
        f"range: {min_id}..{max_id}"
    )


def get_batch(data):

    max_start = (
        len(data)
        - BLOCK_SIZE
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (BATCH_SIZE,),
    )

    x = torch.stack(
        [
            data[i : i + BLOCK_SIZE]
            for i in starts
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1 :
                i + BLOCK_SIZE + 1
            ]
            for i in starts
        ]
    )

    return (
        x.to(
            DEVICE,
            non_blocking=True,
        ),
        y.to(
            DEVICE,
            non_blocking=True,
        ),
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def estimate_loss(model, eval_iters=EVAL_ITERS):

    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data),
    ]:

        losses = []

        for _ in range(eval_iters):

            X, Y = get_batch(data)

            with autocast_context():

                _, loss = model(
                    X,
                    Y,
                )

            losses.append(
                loss.item()
            )

        results[name] = (
            sum(losses)
            / len(losses)
        )

    model.train()

    return results


# ============================================================
# Learning-rate schedule
# ============================================================

def learning_rate(step):

    if step < WARMUP_STEPS:

        return (
            LEARNING_RATE
            * (step + 1)
            / WARMUP_STEPS
        )

    if step >= MAX_STEPS:

        return MIN_LEARNING_RATE

    progress = (
        step - WARMUP_STEPS
    ) / (
        MAX_STEPS - WARMUP_STEPS
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        MIN_LEARNING_RATE
        + cosine
        * (
            LEARNING_RATE
            - MIN_LEARNING_RATE
        )
    )


# ============================================================
# AMP
# ============================================================

def autocast_context():

    if DEVICE.type == "cuda":

        return torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
        )

    return nullcontext()


# ============================================================
# Checkpoint helpers
# ============================================================

def checkpoint_payload(
    model,
    optimizer,
    scaler,
    step,
    best_val,
    vocab_size,
    train_loss=None,
    val_loss=None,
):
    return {
        "step": step,
        "model_state_dict": copy.deepcopy(
            model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler.is_enabled()
            else None
        ),
        "best_val": best_val,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "ffn_hidden": FFN_HIDDEN,
        "seed": SEED,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    step,
    best_val,
    vocab_size,
    train_loss=None,
    val_loss=None,
):
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            best_val=best_val,
            vocab_size=vocab_size,
            train_loss=train_loss,
            val_loss=val_loss,
        ),
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scaler_state = checkpoint.get(
        "scaler_state_dict"
    )

    if (
        scaler.is_enabled()
        and scaler_state
    ):
        scaler.load_state_dict(
            scaler_state
        )

    return (
        int(checkpoint["step"]),
        float(
            checkpoint.get(
                "best_val",
                checkpoint.get(
                    "val_loss",
                    float("inf"),
                ),
            )
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HomeGPT V5."
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help=f"Maximum optimizer steps (default: {MAX_STEPS}).",
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=EVAL_INTERVAL,
        help=f"Evaluate every N steps (default: {EVAL_INTERVAL}).",
    )

    parser.add_argument(
        "--eval-iters",
        type=int,
        default=EVAL_ITERS,
        help=f"Validation batches per evaluation (default: {EVAL_ITERS}).",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help=(
            "Stop after this many consecutive evaluations without "
            "validation improvement (default: "
            f"{EARLY_STOP_PATIENCE})."
        ),
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const=str(
            CHECKPOINT_DIR / "homegpt_v5_latest.pt"
        ),
        default=None,
        help=(
            "Resume from a checkpoint. If no path is supplied, use "
            "checkpoints/homegpt_v5_latest.pt."
        ),
    )

    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Keep torch.compile disabled.",
    )

    return parser.parse_args()


# ============================================================
# Training
# ============================================================

def main():

    args = parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0.")

    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be > 0.")

    if args.eval_iters <= 0:
        raise ValueError("--eval-iters must be > 0.")

    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Missing tokenizer vocabulary: "
            f"{VOCAB_PATH}"
        )

    vocab = json.loads(
        VOCAB_PATH.read_text(
            encoding="utf-8"
        )
    )

    vocab_size = len(vocab)

    if vocab_size <= 0:
        raise ValueError("Vocabulary is empty.")

    validate_dataset(
        train_data,
        "train",
        vocab_size,
    )
    validate_dataset(
        val_data,
        "val",
        vocab_size,
    )

    print("=" * 70)
    print("HOMEGPT V5")
    print("=" * 70)

    print(
        f"Device        : {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU           :",
            torch.cuda.get_device_name(0),
        )

        print(
            "AMP dtype     :",
            AMP_DTYPE,
        )

    print(
        f"Vocabulary    : {vocab_size:,}"
    )

    print(
        f"Context       : {BLOCK_SIZE}"
    )

    print(
        f"Embedding     : {N_EMBD}"
    )

    print(
        f"Layers        : {N_LAYER}"
    )

    print(
        f"Heads         : {N_HEAD}"
    )

    print(
        f"FFN hidden    : {FFN_HIDDEN}"
    )

    print(
        f"Batch         : {BATCH_SIZE}"
    )

    print(
        f"Grad accum    : {GRAD_ACCUM_STEPS}"
    )

    print(
        f"Effective batch: "
        f"{BATCH_SIZE * GRAD_ACCUM_STEPS}"
    )

    print(
        f"Training steps: {args.max_steps}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HomeGPT(
        vocab_size
    ).to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters    : "
        f"{parameter_count / 1e6:.2f}M"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
        fused=(
            DEVICE.type == "cuda"
        ),
    )

    # GradScaler is only needed for FP16.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
            and AMP_DTYPE
            == torch.float16
        ),
    )

    # --------------------------------------------------------
    # Optional torch.compile
    # --------------------------------------------------------

    if (
        DEVICE.type == "cuda"
        and hasattr(torch, "compile")
    ):
        print()
        print("torch.compile is available.")

        if not args.no_compile:
            print("torch.compile requested.")
            model = torch.compile(model)
        else:
            print("torch.compile disabled.")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_val = float("inf")
    start_step = 0
    no_improve_evals = 0

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if args.resume:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        start_step, best_val = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scaler,
        )

        # Resume means "continue after the saved optimizer step."
        start_step += 1

        print()
        print(
            f"✓ Resumed from {resume_path}"
        )
        print(
            f"  next step : {start_step}"
        )
        print(
            f"  best val  : {best_val:.4f}"
        )

    if start_step >= args.max_steps:
        print(
            "Checkpoint is already at or beyond "
            "--max-steps; nothing to train."
        )
        return

    start_time = time.perf_counter()

    model.train()

    for step in range(
        start_step,
        args.max_steps
    ):

        lr = learning_rate(step)

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(
            set_to_none=True
        )

        accumulated_loss = 0.0

        for micro_step in range(
            GRAD_ACCUM_STEPS
        ):

            X, Y = get_batch(
                train_data
            )

            with autocast_context():

                _, loss = model(
                    X,
                    Y,
                )

                loss = (
                    loss
                    / GRAD_ACCUM_STEPS
                )

            accumulated_loss += (
                loss.detach().item()
            )

            if scaler.is_enabled():

                scaler.scale(
                    loss
                ).backward()

            else:

                loss.backward()

        # Unscale before clipping.
        if scaler.is_enabled():

            scaler.unscale_(
                optimizer
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        if scaler.is_enabled():

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            optimizer.step()

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if (
            step % args.eval_interval == 0
            or step == args.max_steps - 1
        ):

            losses = estimate_loss(
                model,
                eval_iters=args.eval_iters,
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            processed_tokens = (
                (step - start_step + 1)
                * BATCH_SIZE
                * GRAD_ACCUM_STEPS
                * BLOCK_SIZE
            )

            tokens_per_sec = (
                processed_tokens
                / max(elapsed_time, 1e-9)
            )

            print(
                f"\nstep {step:6d} | "
                f"lr {lr:.2e} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"{tokens_per_sec / 1000:.1f}k tok/s | "
                f"time {elapsed_time / 60:.1f}m"
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if losses["val"] < best_val:

                best_val = losses["val"]
                no_improve_evals = 0

                save_checkpoint(
                    CHECKPOINT_DIR
                    / "homegpt_v5_best.pt",
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_val,
                    vocab_size,
                    train_loss=losses["train"],
                    val_loss=losses["val"],
                )

                print(
                    "✓ New best checkpoint saved"
                )

            else:
                no_improve_evals += 1

                print(
                    f"  no validation improvement: "
                    f"{no_improve_evals}/{args.patience}"
                )

                if (
                    args.patience > 0
                    and no_improve_evals >= args.patience
                ):
                    print()
                    print(
                        "Early stopping: validation has not "
                        "improved within the configured patience."
                    )
                    break

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if (
            step > 0
            and step % SAVE_INTERVAL == 0
        ):

            save_checkpoint(
                CHECKPOINT_DIR
                / "homegpt_v5_latest.pt",
                model,
                optimizer,
                scaler,
                step,
                best_val,
                vocab_size,
            )

            print(
                "✓ Latest checkpoint saved"
            )

    final_step = step

    save_checkpoint(
        CHECKPOINT_DIR / "homegpt_v5_latest.pt",
        model,
        optimizer,
        scaler,
        final_step,
        best_val,
        vocab_size,
    )

    print("✓ Latest checkpoint saved")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation loss:",
        best_val,
    )

    print(
        "Best checkpoint:",
        CHECKPOINT_DIR
        / "homegpt_v5_best.pt",
    )


if __name__ == "__main__":
    main()
