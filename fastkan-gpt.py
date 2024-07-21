# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan import FastKANLayer, AttentionWithFastKANTransform
from datasets import load_dataset

# Hyperparameters
batch_size = 16  # Number of sequences processed in parallel
n_head = 8       # Number of attention heads in each block
n_embd = 40      # Embedding dimension
n_layer = 4      # Number of transformer blocks
block_size = 32  # Maximum context length
max_iters = 1000 # Maximum number of training iterations
eval_interval = 100  # Evaluate model every 100 iterations
eval_iters = 200     # Number of iterations for evaluation
learning_rate = 1e-3 # Learning rate for optimizer
dropout = 0.0        # Dropout rate (0.0 means no dropout)

# %%
# Load dataset
# text = load_dataset("afmck/text8-chunked1024")['test']
with open('input.txt', 'r') as f:
    text = f.read()

# %%
# Set up device (CPU, CUDA, or MPS)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
print("Using device:", device)

# %%
# Ensure reproducibility
torch.manual_seed(1337)

# %%
# Create character-level tokenizer
stoi = {ch: i for i, ch in enumerate(sorted(list(set(text))))}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

# Encoding and decoding functions
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# %%
# Split data into train and test sets
train_portion = int(0.9 * len(text))
train_data = torch.tensor(encode(text[:train_portion]), dtype=torch.long)
test_data = torch.tensor(encode(text[train_portion:]), dtype=torch.long)

# %%
def get_batch(split):
    """
    Generate a batch of data for training or evaluation.
    
    Args:
        split (str): 'train' or 'val' to specify which dataset to use
    
    Returns:
        tuple: Input tensor (x) and target tensor (y)
    """
    data = train_data if split == 'train' else test_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

# %%
@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss for both training and validation sets.
    
    Returns:
        dict: Contains average loss for 'train' and 'val' splits
    """
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

class FeedForward(nn.Module):
    """Feedforward neural network layer"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = AttentionWithFastKANTransform(
            q_dim=n_embd,
            k_dim=n_embd,
            v_dim=n_embd,
            head_dim=head_size,
            num_heads=n_head,
            gating=True
        )
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        x = x + self.sa(self.ln1(q), self.ln1(k), self.ln1(v))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """Bigram Language Model with transformer architecture"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.embedding(idx)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embedding + pos_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """Generate new tokens based on the given context"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.concat([idx, idx_next], dim=1)
        
        return idx

# %%
# Initialize model and optimizer
model = BigramLanguageModel()
m = model.to(device)

# Print number of parameters
print(sum([p.numel() for p in m.parameters()]) / 1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%
# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses_train_val = estimate_loss()
        print(f"step {iter}: train loss {losses_train_val['train']:.4f}, val loss {losses_train_val['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
# Generate sample text
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))