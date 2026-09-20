import json

with open("combined.txt", encoding="utf-8", errors="replace") as f:
    tokens = f.read()

vocab = []
for i in range(len(tokens)):
    vocab.append(tokens[i])

vocab = list(sorted(set(vocab)))


def count_pairs(tokens):
    pair_counts = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair not in pair_counts:
            pair_counts[pair] = 1
        else:
            pair_counts[pair] += 1

    return pair_counts


def max_count_pair(pair):
    max_count = float("-inf")
    max_pair = None
    for key, value in pair.items():
        if value > max_count:
            max_count = value
            max_pair = key
    return max_pair, max_count


def merge_tokens(tokens, pair_to_merge):
    new_token = []
    i = 0
    merged_pair = pair_to_merge[0] + pair_to_merge[1]
    while i < len(tokens) - 1:
        pair = (tokens[i], tokens[i + 1])
        if pair == pair_to_merge:
            new_token.append(merged_pair)
            i += 2
        else:
            new_token.append(tokens[i])
            i += 1

    if i == len(tokens) - 1:
        new_token.append(tokens[i])

    return new_token, merged_pair


def encode(text, merged_rules):
    tokens = list(text)
    for pair, _ in merged_rules:
        tokens, _ = merge_tokens(tokens, pair)

    return tokens


def decode(ids, itos):
    tokens = ids_to_tokens(ids, itos)
    return "".join(tokens)


merged_rules = []


for merge in range(1000):
    pair_counts = count_pairs(tokens)
    new_max, count = max_count_pair(pair_counts)

    if new_max is None:
        break
    if count <= 1:
        break

    new_tokens, merged_pair = merge_tokens(tokens, new_max)

    merged_rules.append((new_max, merged_pair))

    # print(f"Merge {merge + 1}")
    # print(f"Best Pair: {new_max}")
    # print(f"Frequency: {count}")
    # print(f"Created Token: {merged_pair}")
    # print(f"Number of Tokens: {len(new_tokens)}")
    # print("-" * 40)
    #
    tokens = new_tokens


for _, pair in merged_rules:
    vocab.append(pair)
vocab = list(dict.fromkeys(vocab))
stoi = {ch: i for i, ch in enumerate(vocab)}

itos = {i: ch for i, ch in enumerate(vocab)}


def tokens_to_ids(tokens, stoi):
    ids = []
    for token in tokens:
        ids.append(stoi[token])

    return ids


def ids_to_tokens(ids, itos):
    plain_text = []
    for id in ids:
        plain_text.append(itos[id])

    return plain_text


text = "The Wizard of Oz"
encoded = encode(text, merged_rules)
ids = tokens_to_ids(encoded, stoi)
decoded = decode(ids, itos)
print(decoded)

with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=4)
print("Saved vocabulary!")

with open("merges.json", "w") as f:
    json.dump(merged_rules, f, indent=4)
print("Saved Merged Rules!")
