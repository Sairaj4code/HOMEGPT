import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import encode, decode, tokens_to_ids

# ==========================================================
# CONFIG
# ==========================================================

CHECKPOINT = "homegpt_best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 64
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

# ==========================================================
# LOAD TOKENIZER
# ==========================================================

with open("vocab.json") as f:
    vocab = json.load(f)

with open("merges.json") as f:
    merged_rules = json.load(f)

merged_rules = [(tuple(pair), merged) for pair, merged in merged_rules]

vocab_size = len(vocab)

stoi = {token: i for i, token in enumerate(vocab)}
itos = {i: token for i, token in enumerate(vocab)}

# ==========================================================
# MODEL
# ==========================================================


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x


class GPTLangModule(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index):

        B, T = index.shape

        tok_emb = self.token_embedding_table(index)

        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ==========================================================
# LOAD MODEL
# ==========================================================

model = GPTLangModule(vocab_size).to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device)

if checkpoint["vocab_size"] != vocab_size:
    raise ValueError(
        f"Vocabulary mismatch!\n"
        f"Checkpoint: {checkpoint['vocab_size']}\n"
        f"Tokenizer : {vocab_size}"
    )

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("=" * 60)
print("HomeGPT Loaded!")
print(f"Checkpoint : {CHECKPOINT}")
print(f"Step       : {checkpoint['step']}")
print(f"Val Loss   : {checkpoint['val_loss']:.4f}")
print("=" * 60)

# ==========================================================
# CHAT LOOP
# ==========================================================

while True:
    prompt = input("\nPrompt ('exit' to quit): ")

    if prompt.lower() == "exit":
        break

    if prompt == "":
        context = torch.zeros(
            (1, 1),
            dtype=torch.long,
            device=device,
        )
    else:
        prompt_tokens = encode(prompt, merged_rules)
        prompt_ids = tokens_to_ids(prompt_tokens, stoi)

        context = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

    generated = model.generate(
        context,
        max_new_tokens=500,
        temperature=0.8,
    )

    generated_ids = generated[0].tolist()

    print("\n" + "=" * 60)
    print(decode(generated_ids, itos))
    print("=" * 60)
