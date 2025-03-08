from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define the new text
text = """The sky was a deep shade of blue, and the clouds hung lazily in the air. The sun was setting, casting a warm golden hue across the landscape. The gentle breeze rustled the leaves on the trees, and the world seemed at peace. Birds flew across the horizon, their wings stretched wide as they soared into the distance. It was a perfect evening, full of promise and tranquility."""

# Tokenize the text
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Number of tokens:", len(tokens))

# Load the pre-trained GPT-2 model
model = TFAutoModel.from_pretrained("gpt2")

# Get embeddings for the tokens
inputs = tokenizer(text, return_tensors="tf")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state.numpy()

# Select embeddings for the first 10 tokens
token_embeddings = embeddings[0][:]

print("Shape of embeddings:", token_embeddings.shape)  # Should be (10, embedding_dim)

# Reduce dimensionality using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(token_embeddings)

# Plot the embeddings
plt.figure(figsize=(16, 12))
plt.scatter(reduced_embeddings[:20, 0], reduced_embeddings[:20, 1])  # Show first 20 tokens for better clarity
for i, token in enumerate(tokens[:20]):
    plt.annotate(token.removeprefix('Ġ'), (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
plt.title("2D Visualization of Token Embeddings (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
