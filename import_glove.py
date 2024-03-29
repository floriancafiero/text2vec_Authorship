import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

# Step 1: Loading GloVe embeddings
def load_glove_embeddings(path):
    """
    Load GloVe embeddings from a file and return a dictionary mapping words to vectors.
    """
    embeddings_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# Example: Load GloVe embeddings (update the path to your GloVe file location)
# glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')

# Step 2: Finding similar words (cosine similarity).
def find_similar_words(embeddings_dict, word, topn=10):
    """
    Find and return the most similar words to a given word based on cosine similarity.
    """
    if word not in embeddings_dict:
        return None
    similarities = {}
    target_embedding = embeddings_dict[word]
    for other_word, other_embedding in embeddings_dict.items():
        similarity = 1 - spatial.distance.cosine(target_embedding, other_embedding)
        similarities[other_word] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities[1:topn+1]  # Exclude the first word (itself)

# Example: Find words similar to 'Thionville' (ensure 'Thionville' is in your GloVe vocabulary)
# similar_words = find_similar_words(glove_embeddings, 'Thionville')
# Default of topn as been set up to 10, but should be adjusted depending on the corpus, on the number
# of words examined in the authorship attribution taks, and on the distance measure used. Values outside
# [3 ; 100] are however unlikely to be helpful (cf. Eder, 2022).
