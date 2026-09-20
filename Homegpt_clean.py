import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from tokenizer import encode, decode, tokens_to_ids


DATASET_PATH = "combined.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

block_size = 64
batch_size = 128

max_iters = 10000
learning_rate = 3e-4
eval_iters = 100
eval_interval = 500

n_embd = 384
n_head = 4
n_layer = 4

dropout = 0.2

with open("vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)

with open("merges.json", encoding="utf-8") as f:
    merged_rules = json.load(f)

merged_rules = [(tuple(pair), merges) for pair, merges in merged_rules]

chars = ""
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    text = f.read()


vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


tokens = encode(text, merged_rules)
ids = tokens_to_ids(tokens, stoi)

print(tokens[:20])
print(ids[:20])
print(len(vocab))
data = torch.tensor(ids, dtype=torch.long)

print(data.shape)
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    xi = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in xi])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in xi])
    return x.to(device), y.to(device)


x, y = get_batch("train")

print("Inputs: ")
print(x)
print("Targets: ")
print(y)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


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
        y = self.sa(x)
        x = self.ln1(
            x + y
        )  # <- Keep the previous Knowledge x and build on top it with y
        y = self.ffwd(x)
        x = self.ln2(x + y)  # <- Same as above

        return x


class GPTLangModule(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_crop = index[:, -block_size:]
            logits, _ = self.forward(index_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index


model = GPTLangModule(vocab_size)
m = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist(), itos)
print(f"""
Generated Characters: 
{generated_chars}""")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_val_loss = float("inf")
best_step = 0
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses["train"].item()
        val_loss = losses["val"].item()

        print(
            f"step: {iter}, Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}"
        )

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = iter

            torch.save(
                {
                    "step": iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "vocab_size": vocab_size,
                    "block_size": block_size,
                    "n_embd": n_embd,
                    "n_head": n_head,
                    "n_layer": n_layer,
                    "dropout": dropout,
                },
                "homegpt_best.pt",
            )

            print(f"✓ New best model saved! (step {iter}, val loss {val_loss:.4f})")

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist(), itos)
print(f"""
Generated Characters: 
{generated_chars}""")
