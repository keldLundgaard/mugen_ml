import pickle
import os
import logging

import pandas as pd

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)

DATA_DIR = "/n4Ta/sc_webapp_data/"

info_path = DATA_DIR + "df_with_mp3_info.tsv.gz"
if os.path.exists(info_path):
    w_info_df = pd.read_csv(info_path, sep="\t", compression="gzip")

tfidf_path = DATA_DIR + "tfidf_feats.pckl"
if os.path.exists(tfidf_path):
    logger.info("Loading tfidf features...")
    tfidf_features = pickle.load(open(tfidf_path, "rb"))
    logger.info("Done")
else:
    logger.info("Creating tfidf features")
    tfidf_features = dict()
    for k in ["filename", "title", "artist"]:
        corpus = w_info_df[k].fillna("").values
        vectorizer = TfidfVectorizer()
        tfidf_features["X_" + k] = vectorizer.fit_transform(corpus)
        tfidf_features["vectorizer_" + k] = vectorizer
    pickle.dump(tfidf_features, open(tfidf_path, "wb"))


def search(query=None, category=None, top_k=50, default_category="filename"):
    """
    Search function using TF-IDF features to find top_k items.

    Args:
    query (str): The query string to search for.
    category (str): The category to search within. Must be 'filename', 'title', or 'artist'.
    top_k (int): The number of top results to return.

    Returns:
    list: Indices of the top_k items based on the search query.
    """
    valid_categories = {"filename", "title", "artist"}

    if category is None:
        query, category = parse_query_find(query, valid_categories)
    else:
        if category not in valid_categories:
            raise ValueError(f"Invalid category. Expected one of {valid_categories}")

    if category is None:
        category = default_category

    if query is None:
        raise ValueError("Query must not be None")

    # Transform the query using the appropriate vectorizer
    vectorizer = tfidf_features[f"vectorizer_{category}"]
    query_vector = vectorizer.transform([query])

    # Compute the dot product with the TF-IDF matrix and get the dense output
    tfidf_matrix = tfidf_features[f"X_{category}"]
    scores = np.array(tfidf_matrix.dot(query_vector.T).T.todense())[0]

    # Get indices of the top_k items
    idx_top_k = np.argsort(scores)[::-1][:top_k]

    for i, arg_top in enumerate(idx_top_k):
        if not scores[arg_top] > 0:
            idx_top_k = idx_top_k[:i]
            break

    return idx_top_k


def parse_query_find(input_text, valid_categories):
    category = None
    query = input_text

    for cat in valid_categories:
        cat_pos = input_text.find(cat + ":")
        if cat_pos != -1:
            end_pos = input_text.find(",", cat_pos)
            if end_pos == -1:
                end_pos = len(input_text)
            category = cat
            query = input_text[cat_pos + len(cat) + 1 : end_pos].strip()
            break

    return query, category
