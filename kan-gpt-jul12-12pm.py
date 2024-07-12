# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastkan import FastKANLayer, AttentionWithFastKANTransform
from datasets import load_dataset

# text = load_dataset("afmck/text8-chunked1024")
# %%
# hyperparameters
batch_size = 16 # context length
n_head = 8 # number of heads in each block
n_embd = 40 # embedding dimension
# n_head = 10 # number of heads in each block
# n_embd = 50 # embedding dimension
# ~added later~
block_size = 32 # how many independent seq in parallel?
max_iters = 1000
eval_interval = 100
eval_iters = 200
learning_rate = 1e-3
n_layer = 4 # number of blocks to use
# n_layer = 5 # number of blocks to use
dropout = 0.0 # portion of neurons to turn off

# %%
# read file
with open('../input.txt', 'r') as f:
    text = f.read()

# %%
# gpu stuff
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
print("using device:", device)

# %%
# ensure reproducibility
torch.manual_seed(1337)

# %%
# encode/decode
stoi = {ch: i for i, ch in enumerate(sorted(list(set(text))))}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# %%
# train/test split
train_portion = int(0.9 * len(text))
train_data = torch.tensor(encode(text[:train_portion]), dtype=torch.long)
test_data = torch.tensor(encode(text[train_portion:]), dtype=torch.long)

# %%
# data loading
def get_batch(split):
    data = train_data if split == 'train' else test_data
    idx = torch.randint(len(data) - block_size, (batch_size,)) # MODIFIED: this is data, not text
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y

# %%
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# %%
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # MODIFIED: this is an attribute and not module
        q,k,v = self.query(x), self.key(x), self.value(x)

        wei = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # MODIFIED: (B, T, T); :T ensures that the mask accounts for the fact that there can be less than max number of elems in time dimension
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # ???
        # self.proj = FastKANLayer(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(), # MODIFIED: forgot to add activation in betn
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head # ??? n_embd used in multi headed attention as well whereas
        self.sa = MultiHeadedAttention(n_head, head_size)
        # self.sa = AttentionWithFastKANTransform(
        #     q_dim=n_embd,
        #     k_dim=n_embd,
        #     v_dim=n_embd,
        #     head_dim=head_size,
        #     num_heads=n_head,
        #     gating=True  # You can experiment with this parameter
        # )
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.lm_head = FastKANLayer(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B,T = idx.shape

        token_embedding = self.embedding(idx)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device)) # ???; why time arange
        x = token_embedding + pos_embedding
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # softmax at channel dimension
            idx_next = torch.multinomial(probs, num_samples=1) # predict next token
            idx = torch.concat([idx, idx_next], dim=1)
        
        return idx
    
    # def get_regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    #     reg_loss = 0
    #     for module in self.modules():
    #         if isinstance(module, KANLinear):
    #             reg_loss += module.regularization_loss(regularize_activation, regularize_entropy)
    #     return reg_loss

# %%
model = BigramLanguageModel()
m = model.to(device)

# print no. of parameters
print(sum([p.numel() for p in m.parameters()]) / 1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%
for iter in range(max_iters):
    if iter % eval_interval or iter == max_iters - 1:
        losses_train_val = estimate_loss()
        print(f"step {iter}: train loss {losses_train_val['train']:.4f}, val loss {losses_train_val['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))


