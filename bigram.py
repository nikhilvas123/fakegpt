import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32 # How many independent sequences will we process in parallel?
block_size = 8 # What is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
#device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get unique chars as a sorted list in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
#Char tokenizer - mapping chars to integers uniquely
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Splitting into train and validation set
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    #Generate small batches of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #Generate batchsize number of random offsets
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Context Manager - telling pytorch, that we wont call .backward(), thus we will be efficient
# with memory use since it wont store intermediate values.
@torch.no_grad()
def estimate_loss():
    out = {}
    # Go into evaluation phase. Currently does not matter
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Go into training phase
    model.train()
    return out


# Bigram Model

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and tagets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C) For each value in idx we index the table and get that row
        # For example for 24, we get the 24th row in the embedding table. (B, T, C) = (4, 8, 64)
        # Logits are the score for the next character in the sequence. We are predicting what comes next based on
        # individual identity of a single token (not correlating other characters), 
        # because some characters follow other characters.

        if(targets == None):
            loss = None
        else:    
            B, T, C = logits.shape
            # Reshape for cross_entropy function
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        
        return logits, loss

    # The job of generate is to take idx (B, T) to generate (B, T + 1) .. upto max_new_tokens (B, T +  max_new_tokens)
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits [:, -1, :] # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, C)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)
        
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create pytorch optimizer
# 1e-4 for big models, since our model is small we are using 1e-2
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    
    # Every once in a while evaluate loss on trian and validation sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    #Sample a batch of data
    xb, yb = get_batch('train')
    
    #Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))