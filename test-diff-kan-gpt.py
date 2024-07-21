import torch
import torch.nn as nn
from torch.nn import functional as F

from wavkan.KAN import KANLinear as WaveletKANLinear
from kan.KANLayer import KANLayer as PyKANLayer
from fastkan import FastKANLayer

import matplotlib.pyplot as plt

# hyperparameters
batch_size = 8
block_size = 16
max_iters = 10
eval_interval = max_iters // 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 8
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

# Load and preprocess data
with open('./datasets/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def dynamic_layer(in_dim: int, out_dim: int, id: str, config, bias=None):
    layer_type = config.get(id, 'normal')
    
    if layer_type == 'pykan':
        return PyKANLayer(in_dim=in_dim, out_dim=out_dim, num=5, k=3, device=device)
    elif layer_type == 'wavkan':
        return WaveletKANLinear(in_features=in_dim, out_features=out_dim)
    elif layer_type == 'fastkan':
        return FastKANLayer(input_dim=in_dim, output_dim=out_dim)
    else:
        return nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key = dynamic_layer(n_embd, head_size, bias=False, id='key', config=config)
        self.query = dynamic_layer(n_embd, head_size, bias=False, id='query', config=config)
        self.value = dynamic_layer(n_embd, head_size, bias=False, id='value', config=config)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # Handle both KAN and regular linear layer outputs
        k = k[0] if isinstance(k, tuple) else k
        q = q[0] if isinstance(q, tuple) else q
        v = v[0] if isinstance(v, tuple) else v

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = dynamic_layer(n_embd, n_embd, id='attn_proj', config=config)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = out[0] if isinstance(out, tuple) else out
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, config):
        super().__init__()
        self.net = nn.Sequential(
            dynamic_layer(n_embd, 4 * n_embd, id='ffwd_up', config=config),
            nn.ReLU(),
            dynamic_layer(4 * n_embd, n_embd, id='ffwd_down', config=config),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.net(x)
        return x[0] if isinstance(x, tuple) else x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, config):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, config)
        self.ffwd = FeedForward(n_embd, config)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, config=config) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = dynamic_layer(n_embd, vocab_size, id='lm_head', config=config)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        logits = logits[0] if isinstance(logits, tuple) else logits

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def train_and_evaluate(config):
    model = BigramLanguageModel(config).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return estimate_loss(model)

configurations = [
    {'key': 'normal', 'query': 'normal', 'value': 'normal', 'attn_proj': 'normal', 'lm_head': 'normal', 'ffwd_up': 'normal', 'ffwd_down': 'normal'},  # Baseline
    {'key': 'fastkan', 'query': 'fastkan', 'value': 'fastkan', 'attn_proj': 'fastkan', 'lm_head': 'fastkan', 'ffwd_up': 'fastkan', 'ffwd_down': 'fastkan'},  # All FastKAN
    # {'key': 'wavkan', 'query': 'wavkan', 'value': 'wavkan', 'attn_proj': 'wavkan', 'lm_head': 'wavkan', 'ffwd_up': 'wavkan', 'ffwd_down': 'wavkan'},  # All WavKAN
    {'key': 'normal', 'query': 'normal', 'value': 'normal', 'attn_proj': 'wavkan', 'lm_head': 'wavkan', 'ffwd_up': 'wavkan', 'ffwd_down': 'wavkan'},  # All WavKAN
    {'key': 'pykan', 'query': 'pykan', 'value': 'pykan', 'attn_proj': 'pykan', 'lm_head': 'pykan', 'ffwd_up': 'pykan', 'ffwd_down': 'pykan'},  # All PyKAN
]

results = []
for config in configurations:
    print(f"Training with configuration: {config}")
    losses = train_and_evaluate(config)
    results.append({
        'config': config,
        'train_loss': losses['train'].item(),
        'val_loss': losses['val'].item()
    })

def print_results(results):
    for result in results:
        print(f"Configuration: {result['config']}")
        print(f"Train Loss: {result['train_loss']:.4f}")
        print(f"Validation Loss: {result['val_loss']:.4f}")
        print()

def plot_results(results):
    config_names = [f"Config {i+1}" for i in range(len(results))]
    train_losses = [r['train_loss'] for r in results]
    val_losses = [r['val_loss'] for r in results]

    x = range(len(results))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, train_losses, width, label='Train Loss')
    ax.bar([i + width for i in x], val_losses, width, label='Validation Loss')

    ax.set_ylabel('Loss')
    ax.set_title('Comparison of Different KAN Configurations')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(config_names)
    ax.legend()

    plt.tight_layout()
    plt.show()

print_results(results)
plot_results(results)

# Generate sample text using the best performing model
best_config = min(results, key=lambda x: x['val_loss'])['config']
best_model = BigramLanguageModel(best_config).to(device)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("Generated text:")
print(decode(best_model.generate(context, max_new_tokens=500)[0].tolist()))