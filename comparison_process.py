# Import necessary libraries for NLP tasks and machine learning
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import numpy as np

# Set the working directory to where the corpus is located
os.chdir("100_english_novels")

# Define a function to load text files and compute word frequencies
def load_corpus_and_compute_frequencies(corpus_dir="corpus"):
    # Generate file paths for all text files in the corpus directory
    file_paths = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    # Load and read each text file
    texts = [open(file_path, encoding='utf-8').read() for file_path in file_paths]
    
    # Initialize CountVectorizer for text tokenization and frequency count
    vectorizer = CountVectorizer(input='content', lowercase=True, stop_words='english')
    # Create a document-term matrix (DTM) from the texts
    dtm = vectorizer.fit_transform(texts)
    
    # Convert DTM to a Pandas DataFrame for easier manipulation
    df_dtm = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Compute absolute and relative word frequencies
    word_frequencies_abs = df_dtm.sum().sort_values(ascending=False)
    word_frequencies_rel = word_frequencies_abs / word_frequencies_abs.sum()
    
    return df_dtm, word_frequencies_abs, word_frequencies_rel

# Load pre-trained word vectors, assuming a file path to the model
word_vectors = KeyedVectors.load("word_embedding_models/100_English_novels_GloVe_100_dimensions.model", mmap='r')

# Function to compute word similarities using pre-trained word vectors
def compute_word_similarities(word_vectors, top_n=1000):
    vocabulary = list(word_vectors.index_to_key)
    word_similarities_all = {}
    
    # For each word in the vocabulary, find its most similar words
    for word in vocabulary:
        similar_words = word_vectors.most_similar(word, topn=top_n)
        word_similarities_all[word] = [w for w, _ in similar_words]
    
    # Convert the similarities data into a DataFrame
    df_similarities = pd.DataFrame.from_dict(word_similarities_all, orient='index')
    # Optionally, remove the word itself from its list of similar words
    df_similarities = df_similarities.iloc[:, 1:]
    
    return df_similarities

# Function to compute relative frequencies using a subset of reference words
def compute_subset_frequencies(dtm_df, word_similarities_df, no_of_similar_words):
    final_frequency_matrix = pd.DataFrame(0, index=dtm_df.index, columns=dtm_df.columns[:no_of_similar_words])
    
    for word in dtm_df.columns[:no_of_similar_words]:
        if word in word_similarities_df.index:
            similar_words = word_similarities_df.loc[word, :no_of_similar_words].dropna()
            words_to_compute = [w for w in similar_words if w in dtm_df.columns]
            if word not in words_to_compute:
                words_to_compute.insert(0, word)
            relevant_occurrences = dtm_df[words_to_compute].sum(axis=1)
            final_frequency_matrix[word] = dtm_df[word] / relevant_occurrences
    
    final_frequency_matrix.fillna(0, inplace=True)
    return final_frequency_matrix

# Function to select training texts for a stratified cross-validation scenario
def pick_training_texts(available_texts):
    classes_all = [text.split('_')[0] for text in available_texts]
    unique, counts = np.unique(classes_all, return_counts=True)
    classes_trainable = {u: c for u, c in zip(unique, counts) if c > 1}
    
    texts_in_training_set = []
    for current_class, _ in classes_trainable.items():
        texts_in_current_class = [text for text in available_texts if text.startswith(current_class)]
        np.random.shuffle(texts_in_current_class)
        texts_in_training_set.append(texts_in_current_class[0])
    
    return texts_in_training_set

# Supervised classification function
def supervised_classification(dtm_df, word_similarities_df, mfw_coverage, semantic_area_coverage):
    collect_results_all_similarity_areas = pd.DataFrame(index=mfw_coverage, columns=semantic_area_coverage)
    
    for surrounding_words in semantic_area_coverage:
        subset_freqs_df = compute_subset_frequencies(dtm_df, word_similarities_df, surrounding_words)
        available_texts = subset_freqs_df.index.tolist()
        texts_in_training_set = pick_training_texts(available_texts)
        
        training_set = subset_freqs_df.loc[texts_in_training_set]
        test_set = subset_freqs_df.drop(texts_in_training_set)
        
        for mfw in mfw_coverage:
            X_train, X_test = training_set.iloc[:, :mfw], test_set.iloc[:, :mfw]
            y_train = [text.split('_')[0] for text in X_train.index]
            y_test = [text.split('_')[0] for text in X_test.index]
            
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, average='weighted')
            collect_results_all_similarity_areas.loc[mfw, surrounding_words] = f1
    
    return collect_results_all_similarity_areas

# Load the corpus, compute word frequencies, and prepare data for classification
df_dtm, word_frequencies_abs, word_frequencies_rel = load_corpus_and_compute_frequencies()
df_similarities = compute_word_similarities(word_vectors)

# Specify the coverage of most frequent words and semantic areas for the classification
mfw_coverage = range(100, 1001, 50)
semantic_area_coverage = list(range(1, 10)) + list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10001, 1000))

# Perform supervised classification
# results_df = supervised_classification(df_dtm, df_similarities, mfw_coverage, semantic_area_coverage)
# Optionally, save the results as .csv.
# results_df.to_csv("performance_classification.csv")
