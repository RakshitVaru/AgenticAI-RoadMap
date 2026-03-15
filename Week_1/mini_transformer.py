import numpy as np

np.random.seed(42)

# ── 1. VOCABULARY & TOKENIZATION ──────────────────────────────────────────────

vocab = ["<pad>", "the", "cat", "sat", "on", "mat", "dog", "ran"]
token_to_id = {w: i for i, w in enumerate(vocab)}
id_to_token = {i: w for i, w in enumerate(vocab)}

def tokenize(sentence):
    return [token_to_id[w] for w in sentence.split()]

print("=" * 55)
print("1. TOKENIZATION")
print("=" * 55)
sentence = "the cat sat on the mat"
ids = tokenize(sentence)
print(f"  Text   : '{sentence}'")
print(f"  IDs    : {ids}")
print(f"  Tokens : {[id_to_token[i] for i in ids]}")


# ── 2. TOKEN EMBEDDINGS ───────────────────────────────────────────────────────

VOCAB_SIZE = len(vocab)
D_MODEL = 8   # embedding dimension (tiny — real models use 768–12288)

# Each row = one token's embedding vector (learned during training)
embedding_matrix = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.1

def embed(token_ids):
    return embedding_matrix[token_ids]  # shape: (seq_len, d_model)

print("\n" + "=" * 55)
print("2. TOKEN EMBEDDINGS")
print("=" * 55)
token_embeds = embed(ids)
print(f"  Embedding matrix shape : {embedding_matrix.shape}  (vocab x d_model)")
print(f"  'cat' (id={token_to_id['cat']}) embedding : {embedding_matrix[token_to_id['cat']].round(3)}")
print(f"  Input sequence shape   : {token_embeds.shape}  (seq_len x d_model)")


# ── 3. POSITIONAL ENCODING (sinusoidal, original transformer style) ───────────

def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i]   = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i+1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

print("\n" + "=" * 55)
print("3. POSITIONAL ENCODING")
print("=" * 55)
seq_len = len(ids)
pos_enc = positional_encoding(seq_len, D_MODEL)
x = token_embeds + pos_enc   # combined input to transformer
print(f"  Pos encoding shape : {pos_enc.shape}")
print(f"  Position 0 encoding: {pos_enc[0].round(3)}")
print(f"  Position 1 encoding: {pos_enc[1].round(3)}")
print(f"  (Different position = different encoding)")


# ── 4. SELF-ATTENTION (single head) ──────────────────────────────────────────

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def self_attention(X, mask=False):
    """
    X shape: (seq_len, d_model)
    Returns: (seq_len, d_model) — context-enriched vectors
    """
    d_k = X.shape[1]

    # Learned projection matrices (random here; trained in real models)
    W_Q = np.random.randn(d_k, d_k) * 0.1
    W_K = np.random.randn(d_k, d_k) * 0.1
    W_V = np.random.randn(d_k, d_k) * 0.1

    Q = X @ W_Q   # queries
    K = X @ W_K   # keys
    V = X @ W_V   # values

    # Dot-product attention scores
    scores = Q @ K.T / np.sqrt(d_k)   # shape: (seq_len, seq_len)

    # Optional causal mask (prevents tokens seeing future tokens)
    if mask:
        causal = np.triu(np.ones_like(scores) * -1e9, k=1)
        scores += causal

    weights = softmax(scores)   # attention weights, sum to 1 per row
    output  = weights @ V       # weighted sum of values

    return output, weights

print("\n" + "=" * 55)
print("4. SELF-ATTENTION")
print("=" * 55)
output, attn_weights = self_attention(x, mask=True)
print(f"  Input shape  : {x.shape}")
print(f"  Output shape : {output.shape}  (same — vectors enriched with context)")
print()
print("  Attention weights (each row = one token attending to all others):")
print("  Tokens:", [id_to_token[i] for i in ids])
for i, row in enumerate(attn_weights):
    token = id_to_token[ids[i]]
    vals  = " ".join(f"{v:.2f}" for v in row)
    print(f"    '{token:4s}' attends: [{vals}]")
print("  (causal mask: zeros in upper triangle — can't see future tokens)")


# ── 5. FEED-FORWARD NETWORK (one transformer layer's second sub-component) ────

def feed_forward(X):
    """Simple 2-layer MLP applied to each token independently."""
    d_ff = D_MODEL * 4   # hidden dim typically 4× d_model
    W1 = np.random.randn(D_MODEL, d_ff)  * 0.1
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, D_MODEL)  * 0.1
    b2 = np.zeros(D_MODEL)
    hidden = np.maximum(0, X @ W1 + b1)   # ReLU
    return hidden @ W2 + b2

print("\n" + "=" * 55)
print("5. FEED-FORWARD NETWORK")
print("=" * 55)
ff_out = feed_forward(output)
print(f"  Expands to d_ff={D_MODEL*4}, then back to d_model={D_MODEL}")
print(f"  Output shape: {ff_out.shape}  (seq_len x d_model, same as input)")


# ── 6. FULL TRANSFORMER LAYER (attention + FF + residuals) ───────────────────

def transformer_layer(X):
    attn_out, weights = self_attention(X, mask=True)
    X = X + attn_out          # residual connection
    X = X + feed_forward(X)   # residual connection
    return X, weights

print("\n" + "=" * 55)
print("6. ONE TRANSFORMER LAYER  (attention + FF + residuals)")
print("=" * 55)
layer_out, _ = transformer_layer(x)
print(f"  Input  shape : {x.shape}")
print(f"  Output shape : {layer_out.shape}")
print("  Residual connections preserve the original signal while adding refinement.")


# ── 7. NEXT-TOKEN PREDICTION (output layer) ───────────────────────────────────

def predict_next(X):
    """Unembedding: project final vector back to vocab size, then softmax."""
    W_out = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.1
    logits = X[-1] @ W_out    # only use the last token's vector
    probs  = softmax(logits)
    return logits, probs

print("\n" + "=" * 55)
print("7. NEXT-TOKEN PREDICTION")
print("=" * 55)
logits, probs = predict_next(layer_out)
top_idx = np.argsort(probs)[::-1][:4]
print(f"  Input: '{sentence}'")
print(f"  Top predictions for next token:")
for idx in top_idx:
    print(f"    '{vocab[idx]:6s}'  prob={probs[idx]:.3f}  logit={logits[idx]:.3f}")
print("  (weights are random here — a trained model would predict 'mat' confidently)")


# ── 8. TEMPERATURE SAMPLING ───────────────────────────────────────────────────

def sample_with_temperature(logits, temperature=1.0):
    scaled = logits / temperature
    probs  = softmax(scaled)
    return probs

print("\n" + "=" * 55)
print("8. TEMPERATURE EFFECT ON DISTRIBUTION")
print("=" * 55)
for temp in [0.1, 1.0, 2.0]:
    p = sample_with_temperature(logits, temp)
    top = np.argsort(p)[::-1][:3]
    summary = ", ".join(f"'{vocab[i]}'={p[i]:.2f}" for i in top)
    print(f"  T={temp:.1f}  →  {summary}")
print("  Low T: peaked (confident). High T: flat (creative/random).")


# ── SUMMARY ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("SUMMARY — shapes through the pipeline")
print("=" * 55)
print(f"  Vocab size        : {VOCAB_SIZE}")
print(f"  d_model           : {D_MODEL}")
print(f"  Sequence length   : {seq_len}")
print(f"  Token IDs         : {ids}")
print(f"  Embeddings        : {token_embeds.shape}")
print(f"  + Positional enc  : {x.shape}")
print(f"  After attention   : {output.shape}")
print(f"  After FF network  : {ff_out.shape}")
print(f"  After transformer : {layer_out.shape}")
print(f"  Logits (vocab)    : {logits.shape}")
print(f"  Probabilities     : {probs.shape}")
print()
print("  Real models: same pipeline, just d_model=4096–12288,")
print("  96–128 layers, and billions of trained parameters.")