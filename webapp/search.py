import pickle
import os
import logging

import pandas as pd

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from options import SONGS_DATA_DF_PATH, SONGS_TFIDF_DICT_PATH

USE_SAMPLE_DATA = True

logger = logging.getLogger(__name__)

EXACT_SEARCH_TAGS = {"genre", "sc_user", "year"}
FULL_TEXT_SEARCH_TAGS = {"title", "artist", "filename"}

if os.path.exists(SONGS_TFIDF_DICT_PATH):
    logger.info("Loading tfidf features...")
    tfidf_features = pickle.load(open(SONGS_TFIDF_DICT_PATH, "rb"))
    logger.info("Done")
else:
    if os.path.exists(SONGS_DATA_DF_PATH):
        w_info_df = pd.read_csv(SONGS_DATA_DF_PATH, sep="\t", compression="gzip")
    logger.info("Creating tfidf features")
    tfidf_features = dict()
    for k in ["filename", "title", "artist"]:
        corpus = w_info_df[k].fillna("").values
        vectorizer = TfidfVectorizer()
        tfidf_features["X_" + k] = vectorizer.fit_transform(corpus)
        tfidf_features["vectorizer_" + k] = vectorizer
    pickle.dump(tfidf_features, open(tfidf_path, "wb"))


def search(query=None, songs_df=None, categories=None, top_k=250):
    """
    Performs an enhanced search on a songs DataFrame using TF-IDF features to 
    find items matching a given query, within specified categories. It supports 
    both exact and full-text searches, allowing users to narrow down the search 
    results to the top 'k' items most relevant to the query.

    The function requires a pre-processed DataFrame where each song's information 
    is indexed and associated with TF-IDF features for efficient text-based 
    searching.

    Args:
        query (str): The query string to search for. This can include both exact 
            matches (enclosed in quotes) and general search terms.
        songs_df (pd.DataFrame): The DataFrame containing songs information. 
            Each row represents a song, and columns should include details such 
            as 'filename', 'title', 'artist', among others specified in 
            'categories'.
        categories (list of str, optional): The categories (columns in 'songs_df') 
            to search within. This must be a subset of the DataFrame's columns. 
            Default searches across ['filename', 'title', 'artist'].
        top_k (int): The number of top results to return. Determines the 'k' 
            highest scoring items based on the search query's relevance. Defaults 
            to 50.

    Returns:
        dict: A dictionary with categories (from 'categories' argument or default 
            set) as keys and lists of indices (from 'songs_df') for the top 'k' 
            items based on the search query as values. The indices correspond to 
            rows in 'songs_df' that best match the search criteria.

    Note:
        The search functionality is heavily reliant on the structure and 
        preprocessing of 'songs_df', including the presence of TF-IDF features. 
        Ensure that 'songs_df' is properly preprocessed with TF-IDF vectors for 
        each searchable category.
    """
    parsed_query = parse_query(query)
    results = {}
    print(parsed_query)

    matches = songs_df.title.apply(lambda x: True).rename("match")
    exact_matching = parsed_query["exact"]

    print(f"before: {sum(matches):,d}")
    if len(exact_matching):
        for tag, tag_query in exact_matching.items():
            matches = matches & (songs_df[tag].str.lower() == tag_query.lower())

    print(f"after: {sum(matches):,d}")
    consolidated_matches = matches[matches]

    full_text = parsed_query["full-text"]
    scores = None
    for tag in FULL_TEXT_SEARCH_TAGS:
        tag_queries = []

        if tag in full_text:
            tag_queries.append(full_text[tag])
        tag_queries.extend(parsed_query["general-full-text"])

        if len(tag_queries): # only need to do anything if elements here
            tfidf_ma = tfidf_features[f"X_{tag}"]
            tfidf_matched_ma = tfidf_ma[matches, :]  # operate in match space
            vectorizer = tfidf_features[f"vectorizer_{tag}"]

            query_vec = vectorizer.transform(tag_queries)
            tag_scores = (
                np.array(tfidf_matched_ma.dot(query_vec.T).T.todense())
                .sum(axis=0)
                )
            scores = tag_scores if scores is None else scores + tag_scores

    if scores is None:
        idx_top_k = list(consolidated_matches.sample(frac=1).index)[:top_k]
    else:
        # Get indices of the top_k items based on highest scores
        m_idx_top_k = np.argsort(scores)[::-1][:top_k]
        for i, arg_top in enumerate(m_idx_top_k):
            if not scores[arg_top] > 0:
                m_idx_top_k = m_idx_top_k[:i]
                break
        # convert back to original index
        idx_top_k = list(consolidated_matches.iloc[m_idx_top_k].index)

    return idx_top_k


def parse_query(query):
    """
    Parses a query string into categorized search directives for efficient
    searching within a large music database. The function supports exact search
    directives (where search terms must exactly match values in the database)
    and full-text search directives (where search terms are matched based on
    text similarity).

    Args:
        query (str): The input query string, which can include directives for
            exact matches (enclosed in single or double quotes) and general
            search terms. Directives can be separated by semicolons (';').
            Short-hand notations for certain search fields (e.g., 'g' for
            'genre', 'a' for 'artist') are also supported.

    Returns:
        dict: A dictionary containing categorized search directives. The
            dictionary includes keys such as 'exact' for exact search
            directives, 'full-text' for full-text search directives, and
            'general-full-text' for general search terms. The values are
            structured to facilitate efficient search implementation.

    Examples:

    Exact Match Search:
    >>> parse_query("artist: 'The Beatles'")
    {'exact': {'artist': 'The Beatles'}, 'full-text': {}, 'general-full-text': []}

    Full-Text Search:
    >>> parse_query("title: Imagine")
    {'exact': {}, 'full-text': {'title': 'Imagine'}, 'general-full-text': []}

    Mixed Search Types:
    >>> parse_query("genre: rock; artist: 'Led Zeppelin'; title: Stairway")
    {'exact': {'artist': 'Led Zeppelin'}, 'full-text': {'title': 'Stairway', 'genre': 'rock'}, 'general-full-text': []}

    Using Short-Hand Notations:
    >>> parse_query("a: 'Pink Floyd'; t: Comfortably Numb")
    {'exact': {'artist': 'Pink Floyd'}, 'full-text': {'title': 'Comfortably Numb'}, 'general-full-text': []}

    General Full-Text Search:
    >>> parse_query("1973")
    {'exact': {}, 'full-text': {}, 'general-full-text': ['1973']}

    Note:
        The parsing logic differentiates between exact and full-text searches
        based on the presence of quotation marks around search terms. It also
        supports the use of short-hand notations for commonly searched fields,
        enhancing the usability of the search function.
    """

    # Define sets of tags for exact search and full-text search

    short_hands = {
        "g": "genre",
        "a": "artist",
        "t": "title",
        "f": "filename",
        "u": "sc_user",
        "y": "year"
    }

    # Split the query into parts
    parts = query.split(";")

    # Initialize output dictionary
    output = {"exact": {}, "full-text": {}, "general-full-text": []}

    # Process each part
    for part in parts:
        part = part.strip()  # Remove leading and trailing whitespace
        if not part:  # Skip empty parts
            continue
        part_splitted = part.split(" ")
        first_word, rest = part_splitted[0], " ".join(part_splitted[1:]).strip()
        first_word = first_word.split(":")[0].strip().lower()

        first_word = short_hands.get(first_word, first_word) # convert with short-hands

        # Normalize quotations for exact match in full-text search
        is_exact = rest.startswith(("'", '"')) and rest.endswith(("'", '"'))
        if is_exact:
            rest = rest[1:-1]  # Remove the enclosing quotes

        # Determine if the tag indicates an exact search or full-text search
        if first_word in EXACT_SEARCH_TAGS or is_exact:
            output["exact"][first_word] = rest
        elif first_word in FULL_TEXT_SEARCH_TAGS:
            output["full-text"][first_word] = rest
        else:
            output["general-full-text"].append(part)

    return output
