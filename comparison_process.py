# Import necessary libraries
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Setting the working directory
os.chdir("1")

# Load and preprocess text files
# For loading and tokenizing text, computing word frequencies, and preparing a document-term matrix
# Using sklearn's CountVectorizer for tokenization and frequency counts

def load_corpus_and_compute_frequencies(corpus_dir="corpus"):
    file_paths = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    texts = [open(file_path, encoding='utf-8').read() for file_path in file_paths]
    
    # Vectorize the text data: tokenization and frequency counts
    vectorizer = CountVectorizer(input='content', lowercase=True, stop_words='english')
    dtm = vectorizer.fit_transform(texts)  # Document-term matrix
    
    # Convert DTM to Pandas DataFrame for easy manipulation
    df_dtm = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Compute absolute frequencies
    word_frequencies_abs = df_dtm.sum().sort_values(ascending=False)
    
    # Compute relative frequencies
    word_frequencies_rel = word_frequencies_abs / word_frequencies_abs.sum()
    
    return df_dtm, word_frequencies_abs, word_frequencies_rel

# Load the corpus and compute word frequencies
df_dtm, word_frequencies_abs, word_frequencies_rel = load_corpus_and_compute_frequencies()

# Load pre-trained word vectors (for example, using Gensim's KeyedVectors)
# This assumes you have a word vector model file (e.g., Word2Vec, GloVe) available in the specified format
# Replace 'word_embedding_model_file' with the path to your model file
word_vectors = KeyedVectors.load("100_english_novels/word_embedding_models/100_English_novels_GloVe_100_dimensions.model", mmap='r')

# Compute word similarities
def compute_word_similarities(word_vectors, top_n=1000):
    vocabulary = list(word_vectors.index_to_key)
    word_similarities_all = {}
    
    for word in vocabulary:
        # Get the most similar words for each word in the vocabulary
        similar_words = word_vectors.most_similar(word, topn=top_n)
        word_similarities_all[word] = [w for w, _ in similar_words]
    
    # Convert to DataFrame for easier manipulation and save
    df_similarities = pd.DataFrame.from_dict(word_similarities_all, orient='index')
    
    # Remove the word itself being the most similar (if applicable)
    df_similarities = df_similarities.iloc[:, 1:]
    
    return df_similarities

# Compute word similarities for the vocabulary
df_similarities = compute_word_similarities(word_vectors)

# Example: Save the DataFrame of word similarities to a file (optional)
# df_similarities.to_csv("ranked_word_similarities.csv")

def compute_subset_frequencies(dtm_df, word_similarities_df, no_of_similar_words):
    """
    Compute relative word frequencies using a subset of reference words.
    
    :param dtm_df: Document-term matrix as a Pandas DataFrame.
    :param word_similarities_df: DataFrame containing most similar words.
    :param no_of_similar_words: Number of similar words to consider.
    :return: DataFrame with improved frequencies.
    """
    final_frequency_matrix = pd.DataFrame(0, index=dtm_df.index, columns=dtm_df.columns[:no_of_similar_words])
    
    for word in dtm_df.columns[:no_of_similar_words]:
        if word in word_similarities_df.index:
            similar_words = word_similarities_df.loc[word, :no_of_similar_words].dropna()
            words_to_compute = [w for w in similar_words if w in dtm_df.columns]
            # Add the current word to ensure it's included in the computation
            if word not in words_to_compute:
                words_to_compute.insert(0, word)
            
            # Compute occurrences of the relevant words
            relevant_occurrences = dtm_df[words_to_compute].sum(axis=1)
            # Compute new relative frequencies
            final_frequency_matrix[word] = dtm_df[word] / relevant_occurrences
    
    # Replace NaN values with 0s
    final_frequency_matrix.fillna(0, inplace=True)
    return final_frequency_matrix

def pick_training_texts(available_texts):
    """
    Select training set texts in a stratified cross-validation scenario.
    
    :param available_texts: List of text filenames.
    :return: List of filenames chosen for the training set.
    """
    classes_all = [text.split('_')[0] for text in available_texts]
    unique, counts = np.unique(classes_all, return_counts=True)
    classes_trainable = {u: c for u, c in zip(unique, counts) if c > 1}
    
    texts_in_training_set = []
    for current_class, _ in classes_trainable.items():
        texts_in_current_class = [text for text in available_texts if text.startswith(current_class)]
        np.random.shuffle(texts_in_current_class)
        # Assuming one text per class for simplicity
        texts_in_training_set.append(texts_in_current_class[0])
    
    return texts_in_training_set

# Example usage
# Assuming dtm_df is the document-term matrix as a DataFrame and df_similarities is the similarities DataFrame
# improved_frequencies = compute_subset_frequencies(dtm_df, df_similarities, 100)

# Assuming available_texts is a list of filenames in your corpus
# training_texts = pick_training_texts(available_texts)



