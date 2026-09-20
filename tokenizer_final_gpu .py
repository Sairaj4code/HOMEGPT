"""
HomeGPT - Large-Corpus GPU BPE Tokenizer
=========================================

This keeps the HomeGPT tokenizer custom and GPU-first, but is designed
for large corpora such as a ~1 GiB FineWeb-Edu text file.

IMPORTANT:
- The downloadable/runtime file name remains: tokenizer_final_gpu.py
- It does NOT use a Hugging Face tokenizer or pretrained model.
- BPE merge learning is performed on a representative sample on CUDA.
- The complete corpus is then encoded with the learned BPE rules in
  streaming chunks, so the 1 GiB raw corpus is never copied to GPU.
- Final artifacts keep the same names expected by HomeGPT training:

    tokenizer/vocab.json
    tokenizer/merges.json
    tokenizer/bpe_checkpoint.json
    data/train_tokens.pt
    data/val_tokens.pt

Default input:
    data/combined.txt

Default target vocabulary:
    4096 tokens, including 7 special tokens.

Recommended first run:
    python3 tokenizer_final_gpu.py --sample-mb 32 --merges 4089

You can benchmark the BPE learner first:
    python3 tokenizer_final_gpu.py --sample-mb 32 --merges 50

Why sample-based BPE?
---------------------
A 1 GiB corpus cannot safely be represented as a GPU int32 tensor on a
4 GiB GPU. Instead, we learn the merge rules from a representative
sample, then encode the full corpus in CPU-streaming chunks.

The sample is spread across the input file rather than taking only the
first N MB, which gives the learned vocabulary exposure to more of the
corpus.

No external tokenizer is required.
"""

import argparse
import heapq
import json
import math
import os
import time
from pathlib import Path

import torch


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/combined.txt")
DATA_DIR = Path("data")
OUTPUT_DIR = Path("tokenizer")

VOCAB_SIZE = 8192

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

SPECIAL_STOI = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

STREAM_CHUNK_MB = 4
OUTPUT_DTYPE = torch.int64


# ============================================================
# Utilities
# ============================================================


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def elapsed(start, device):
    synchronize(device)
    return time.perf_counter() - start


def human_bytes(value):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(value)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


# ============================================================
# Character vocabulary
# ============================================================


def scan_characters(path, chunk_mb=16):
    """
    Stream the corpus and build a Unicode-codepoint vocabulary without
    loading the 1 GiB file into one Python string.
    """
    chars = set()
    chunk_size = chunk_mb * 1024 * 1024

    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chars.update(chunk)

    return sorted(chars)


def build_character_vocab(characters):
    offset = len(SPECIAL_TOKENS)

    stoi = {ch: idx + offset for idx, ch in enumerate(characters)}

    itos = {idx + offset: ch for idx, ch in enumerate(characters)}

    return stoi, itos


# ============================================================
# Representative sample
# ============================================================


def read_spread_sample(path, sample_mb):
    """
    Read several equally-spaced byte regions from the corpus.

    We decode each region independently with errors='replace'. The
    regions are deliberately bounded and do not require the complete
    corpus to exist in RAM.
    """
    size = path.stat().st_size
    target = max(1, sample_mb) * 1024 * 1024

    if size <= target:
        return path.read_text(encoding="utf-8", errors="replace")

    pieces = []
    regions = max(4, min(16, math.ceil(target / (4 * 1024 * 1024))))
    piece_size = max(1, target // regions)

    # Avoid sampling only the beginning of the dataset.
    for i in range(regions):
        fraction = i / max(1, regions - 1)
        offset = int((size - piece_size) * fraction)

        with path.open("rb") as f:
            f.seek(offset)
            raw = f.read(piece_size)

        text = raw.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        pieces.append(text)

    return "\n".join(pieces)


# ============================================================
# GPU BPE learning
# ============================================================


def text_to_gpu_tokens(text, stoi, device):
    ids = [stoi[ch] for ch in text]
    return torch.tensor(ids, dtype=torch.int32, device=device)


@torch.no_grad()
def count_pairs(tokens):
    if tokens.numel() < 2:
        return None, None

    left = tokens[:-1].to(torch.int64)
    right = tokens[1:].to(torch.int64)

    keys = (left << 32) | right

    unique_keys, counts = torch.unique(
        keys,
        return_counts=True,
        sorted=False,
    )

    return unique_keys, counts


@torch.no_grad()
def get_best_pair(tokens):
    keys, counts = count_pairs(tokens)

    if keys is None or counts.numel() == 0:
        return None, 0

    index = torch.argmax(counts)
    key = int(keys[index].item())
    frequency = int(counts[index].item())

    left = key >> 32
    right = key & 0xFFFFFFFF

    return (left, right), frequency


@torch.no_grad()
def merge_pair(tokens, pair, new_token_id):
    first, second = pair

    if tokens.numel() < 2:
        return tokens, 0

    left = tokens[:-1]
    right = tokens[1:]

    selected = (left == first) & (right == second)

    if not torch.any(selected):
        return tokens, 0

    # Greedy left-to-right non-overlapping matches.
    if selected.numel() > 1:
        selected = selected.clone()
        selected[1:] &= ~selected[:-1]

    count = int(selected.sum().item())

    if count == 0:
        return tokens, 0

    remove = torch.zeros(
        tokens.numel(),
        dtype=torch.bool,
        device=tokens.device,
    )
    remove[1:] = selected

    output = tokens[~remove].clone()

    starts = torch.nonzero(
        selected,
        as_tuple=False,
    ).flatten()

    removed_before = torch.cumsum(
        remove.to(torch.int32),
        dim=0,
    )

    output_positions = starts - removed_before[starts]
    output[output_positions] = new_token_id

    return output, count


# ============================================================
# Vocabulary
# ============================================================


def build_vocab(base_itos, merges):
    token_strings = dict(base_itos)

    for pair, token_id in merges:
        left_id, right_id = pair
        token_strings[token_id] = token_strings[left_id] + token_strings[right_id]

    max_id = max(token_strings) if token_strings else -1
    vocab = [""] * (max_id + 1)

    for token, idx in SPECIAL_STOI.items():
        vocab[idx] = token

    for idx, token in token_strings.items():
        vocab[idx] = token

    return vocab


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Exact Python BPE encoder for large streaming corpus
# ============================================================


class FastBPEEncoder:
    """
    Heap-based BPE encoder.

    Unlike the old runtime encoder, this does NOT loop over every merge
    rule for every character. It uses merge rank as a priority and only
    revisits pairs affected by an actual merge.

    This makes full-corpus encoding practical enough to run in chunks.
    """

    def __init__(self, char_stoi, merges, vocab):
        self.char_stoi = char_stoi
        self.vocab = vocab

        # Pair -> merge rank and resulting token ID.
        self.rank = {}
        self.result = {}

        for rank, (pair, token_id) in enumerate(merges):
            key = (int(pair[0]), int(pair[1]))
            self.rank[key] = rank
            self.result[key] = int(token_id)

        self.max_token_chars = max(
            (len(x) for x in vocab if x),
            default=1,
        )

    def encode_chunk(self, text):
        """
        Encode one chunk using a linked-list representation plus a
        min-heap of merge candidates.

        Returns token IDs and the original character span represented
        by each token.
        """
        if not text:
            return [], []

        ids = []
        for ch in text:
            ids.append(self.char_stoi.get(ch, SPECIAL_STOI["<|unk|>"]))

        n = len(ids)
        if n == 0:
            return [], []

        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1

        alive = [True] * n
        spans = [1] * n

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
                heapq.heappush(heap, (rank, left, right))

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
            spans[left] += spans[right]

            alive[right] = False

            after = nxt[right]
            nxt[left] = after

            if after >= 0:
                prev[after] = left

            before = prev[left]
            push_pair(before)
            push_pair(left)

        output_ids = []
        output_spans = []

        i = 0
        while i >= 0 and i < n:
            if alive[i]:
                output_ids.append(ids[i])
                output_spans.append(spans[i])
            i = nxt[i]

        return output_ids, output_spans


# ============================================================
# Streaming full-corpus encoder
# ============================================================


def encode_full_corpus(path, encoder, chunk_mb=4):
    """
    Encode the complete corpus without putting the raw 1 GiB text on
    the GPU.

    A small character carry is retained between chunks. We only emit
    tokens whose source span is safely before the carry boundary.
    """
    chunk_size = chunk_mb * 1024 * 1024

    all_token_ids = []
    carry = ""

    total_chars = 0
    last_report = time.perf_counter()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            total_chars += len(chunk)

            text = carry + chunk
            ids, spans = encoder.encode_chunk(text)

            # Keep enough raw text at the end to protect merges that
            # could cross the next chunk boundary.
            safe_chars = max(
                1,
                encoder.max_token_chars,
            )

            emitted = 0
            covered = 0

            # Find the largest token prefix whose represented raw
            # character span is safely before the end.
            target = max(0, len(text) - safe_chars)

            for span in spans:
                if covered + span > target:
                    break
                covered += span
                emitted += 1

            all_token_ids.extend(ids[:emitted])

            # Reconstruct the un-emitted raw suffix from the source
            # character count. This is small because max_token_chars is
            # normally tiny compared with the chunk.
            emitted_chars = covered
            carry = text[emitted_chars:]

            now = time.perf_counter()
            if now - last_report >= 5:
                print(
                    f"  encoded ~{total_chars:,} chars | "
                    f"tokens emitted {len(all_token_ids):,}"
                )
                last_report = now

    # Final carry: encode it and emit everything.
    if carry:
        ids, _ = encoder.encode_chunk(carry)
        all_token_ids.extend(ids)

    return all_token_ids


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-mb",
        type=int,
        default=32,
        help=(
            "Approximate corpus size used to learn BPE merges. "
            "32 MB is a safe default for a 4 GB GPU."
        ),
    )

    parser.add_argument(
        "--merges",
        type=int,
        default=4089,
        help=(
            "Maximum number of BPE merges. The actual maximum is "
            "4096 - special_tokens - character_vocab_size."
        ),
    )

    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=4,
        help="Streaming encoding chunk size in MB.",
    )

    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("=" * 72)
    print("HOMEGPT LARGE-CORPUS GPU BPE TOKENIZER")
    print("=" * 72)
    print(f"Input       : {DATA_PATH}")
    print(f"Input size  : {human_bytes(DATA_PATH.stat().st_size)}")
    print(f"Device      : {device}")

    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Pass 1: character vocabulary
    # --------------------------------------------------------

    print()
    print("[1/5] Scanning corpus for Unicode characters...")

    start = time.perf_counter()
    characters = scan_characters(DATA_PATH)
    scan_time = time.perf_counter() - start

    base_stoi, base_itos = build_character_vocab(characters)
    base_vocab_size = len(base_stoi)

    print(f"Unique characters: {base_vocab_size:,}")
    print(f"Scan time        : {scan_time:.2f}s")

    maximum_merges = VOCAB_SIZE - len(SPECIAL_TOKENS) - base_vocab_size

    if maximum_merges <= 0:
        raise RuntimeError(
            "Character vocabulary is too large for a 4096-token "
            "vocabulary. Increase VOCAB_SIZE or reduce character set."
        )

    requested_merges = min(
        args.merges,
        maximum_merges,
    )

    print(f"Special tokens   : {len(SPECIAL_TOKENS)}")
    print(f"Target vocab     : {VOCAB_SIZE}")
    print(f"Max BPE merges   : {maximum_merges}")
    print(f"Requested merges : {requested_merges}")

    # --------------------------------------------------------
    # Pass 2: representative sample + GPU BPE learning
    # --------------------------------------------------------

    print()
    print("[2/5] Loading representative BPE-learning sample...")

    sample = read_spread_sample(
        DATA_PATH,
        args.sample_mb,
    )

    print(
        f"Sample characters: {len(sample):,} "
        f"({human_bytes(len(sample.encode('utf-8')))})"
    )

    sample_tokens = text_to_gpu_tokens(
        sample,
        base_stoi,
        device,
    )

    print(f"Sample tokens    : {sample_tokens.numel():,}")

    print()
    print("[3/5] Learning BPE merges on GPU...")

    merges = []

    next_token_id = len(SPECIAL_TOKENS) + base_vocab_size

    total_start = time.perf_counter()

    for merge_index in range(requested_merges):
        pair, frequency = get_best_pair(sample_tokens)

        if pair is None or frequency < 2:
            print("No useful pair remains.")
            break

        merge_start = time.perf_counter()

        sample_tokens, occurrences = merge_pair(
            sample_tokens,
            pair,
            next_token_id,
        )

        merge_time = elapsed(
            merge_start,
            device,
        )

        merges.append((pair, next_token_id))
        next_token_id += 1

        completed = merge_index + 1

        if completed <= 10 or completed % 25 == 0 or completed == requested_merges:
            elapsed_total = elapsed(
                total_start,
                device,
            )
            average = elapsed_total / completed
            remaining = requested_merges - completed
            eta = average * remaining

            print(
                f"merge {completed:4d} | "
                f"pair={pair} | "
                f"freq={frequency:8d} | "
                f"merged={occurrences:8d} | "
                f"sample_tokens={sample_tokens.numel():10,d} | "
                f"merge={merge_time:6.3f}s | "
                f"ETA={eta / 60:6.2f}m"
            )

    total_bpe_time = elapsed(
        total_start,
        device,
    )

    # --------------------------------------------------------
    # Build and save tokenizer artifacts
    # --------------------------------------------------------

    print()
    print("[4/5] Building tokenizer artifacts...")

    vocab = build_vocab(
        base_itos,
        merges,
    )

    if len(vocab) > VOCAB_SIZE:
        raise AssertionError(f"Vocabulary overflow: {len(vocab)} > {VOCAB_SIZE}")

    vocab_path = OUTPUT_DIR / "vocab.json"
    merges_path = OUTPUT_DIR / "merges.json"
    checkpoint_path = OUTPUT_DIR / "bpe_checkpoint.json"

    save_json(vocab_path, vocab)

    save_json(
        merges_path,
        [
            [
                [int(pair[0]), int(pair[1])],
                int(token_id),
            ]
            for pair, token_id in merges
        ],
    )

    save_json(
        checkpoint_path,
        {
            "version": "large-corpus-v1",
            "special_tokens": SPECIAL_TOKENS,
            "target_vocab_size": VOCAB_SIZE,
            "actual_vocab_size": len(vocab),
            "base_vocab_size": base_vocab_size,
            "completed_merges": len(merges),
            "merge_training_sample_mb": args.sample_mb,
            "merge_training_sample_chars": len(sample),
            "input_path": str(DATA_PATH),
        },
    )

    print(f"Saved: {vocab_path}")
    print(f"Saved: {merges_path}")
    print(f"Saved: {checkpoint_path}")

    # --------------------------------------------------------
    # Full-corpus encoding
    # --------------------------------------------------------

    print()
    print("[5/5] Encoding the COMPLETE corpus...")
    print(
        "This stage is streaming and CPU-side; the 1 GiB raw corpus "
        "is NOT copied to GPU."
    )

    encoder = FastBPEEncoder(
        base_stoi,
        merges,
        vocab,
    )

    encode_start = time.perf_counter()

    token_ids = encode_full_corpus(
        DATA_PATH,
        encoder,
        chunk_mb=args.chunk_mb,
    )

    encode_time = time.perf_counter() - encode_start

    if not token_ids:
        raise RuntimeError("Full-corpus encoding produced zero tokens.")

    tokens = torch.tensor(
        token_ids,
        dtype=OUTPUT_DTYPE,
    )

    # --------------------------------------------------------
    # Validate IDs
    # --------------------------------------------------------

    min_id = int(tokens.min().item())
    max_id = int(tokens.max().item())

    if min_id < 0 or max_id >= len(vocab):
        raise RuntimeError(
            f"Token ID range invalid: min={min_id}, max={max_id}, vocab={len(vocab)}"
        )

    split_index = int(tokens.numel() * 0.90)

    train_tokens = tokens[:split_index]
    val_tokens = tokens[split_index:]

    train_path = DATA_DIR / "train_tokens.pt"
    val_path = DATA_DIR / "val_tokens.pt"

    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)

    # --------------------------------------------------------
    # Round-trip test
    # --------------------------------------------------------

    runtime = FastBPEEncoder(
        base_stoi,
        merges,
        vocab,
    )

    sample_text = (
        "Hello HomeGPT! "
        "This is a tokenizer round-trip test. "
        "Python, Linux, transformers, and machine learning."
    )

    sample_ids = runtime.encode_chunk(sample_text)[0]
    decoded = "".join(vocab[i] for i in sample_ids)

    print()
    print("=" * 72)
    print("FINAL VALIDATION")
    print("=" * 72)
    print(f"Vocabulary size      : {len(vocab):,}")
    print(f"BPE merges            : {len(merges):,}")
    print(f"Full corpus tokens    : {tokens.numel():,}")
    print(f"Training tokens       : {train_tokens.numel():,}")
    print(f"Validation tokens     : {val_tokens.numel():,}")
    print(f"Token ID range        : {min_id} .. {max_id}")
    print(f"BPE learning time     : {total_bpe_time / 60:.2f} min")
    print(f"Full encoding time    : {encode_time / 60:.2f} min")
    print(f"Round-trip             : {'PASS' if decoded == sample_text else 'FAIL'}")

    print()
    print("FILES CREATED:")
    print(f"  {vocab_path}")
    print(f"  {merges_path}")
    print(f"  {checkpoint_path}")
    print(f"  {train_path}")
    print(f"  {val_path}")

    if decoded != sample_text:
        print()
        print("Expected:")
        print(sample_text)
        print()
        print("Decoded:")
        print(decoded)

    print()
    print("DONE. HomeGPT training can now use the fresh tokenized corpus.")


if __name__ == "__main__":
    main()
