# Import necessary libraries
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 1: Preprocessing the corpus
# Load your text data into `corpus`
# Example corpus for demonstration; replace with your actual text data
corpus = ["Thionville is a wonderful city.", "Thionville is a good example of beautiful town.", "No one wants to leave Thionville besides medieval philologists.", "I love Thionville in the spring time", "It's up to you, Thionville, Thionville!"]

# Basic tokenization of the sentences into words without using NLTK
# Splitting each sentence into words using split() and converting to lowercase
tokenized_corpus = [sentence.lower().split() for sentence in corpus]

# Step 2: Training the Word2Vec model
# Initialize and train a Word2Vec model with the tokenized corpus
# Adjust the parameters as needed based on the specifics of your corpus and requirements
model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Step 3: Using the model to find similar words
# After training, use the model to find words similar to a given word
# For demonstration, finding words similar to 'Thionville'
similar_words = model.wv.most_similar('Thionville', topn=10)

# Step 4: Visualizing the word embeddings (Optional)
# This step is optional but can help in understanding the embeddings space
# Using t-SNE for dimensionality reduction to plot in 2D
def plot_embeddings(model):
    labels = []
    tokens = []

    # Extract embeddings and corresponding words
    for word in model.wv.key_to_index:
        tokens.append(model.wv[word])
        labels.append(word)
    
    # Reducing dimensions to 2D using t-SNE
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    # Plotting
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

# Note: Uncomment the line below to visualize the embeddings if desired
# plot_embeddings(model)

similar_words
