# pip install transformers
from  transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

for word in ["Unbelievable", "the cat sat", "2024"]:
    tokens = tok.tokenize(word)
    ids    = tok.encode(word, add_special_tokens=False)
    print(f"{word} → tokens: {tokens} | ids: {ids}")
