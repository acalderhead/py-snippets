#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ─────────────────────────────────────────────────────────────────────────────
# Module Documentation
# ─────────────────────────────────────────────────────────────────────────────

"""
create_bertopics_model.py
─────────────────────────
Generalized BERTopic modeling pipeline for extracting interpretable topics 
from free-text datasets such as surveys, interview transcripts, or feedback 
responses.

The script is fully parameterized and designed for reuse in analytical projects
requiring text clustering or thematic synthesis. It retains diagnostic 
transparency for iterative tuning and model reproducibility.

This portfolio-safe version contains no project-specific stopwords or 
references.

Typical Workflow
────────────────
1. Load text data (TXT)
2. Clean and normalize text
3. Construct a BERTopic model
4. Fit and optionally reduce topics
5. Label and export results (CSV)

Tuning Overview
───────────────
Top-Level Tuning
    - `min_topic_size`: 
        - ↑ fewer, broader topics
        - ↓ more, finer topics.
    - `n_neighbors` (UMAP): 
        - ↑ smoother global structure
        - ↓ preserves local details.
    - `embedding_model`: tradeoff between semantic precision and runtime cost.

Fine-Tuning
    - `min_samples`: 
        - ↑ conservative clustering (more "noise")
        - ↓ permissive merging.
    - `n_components`: 
        - ↑ retains more variance in UMAP
        - ↓ faster but may oversimplify.
    - `min_df`: 
        - ↑ ignore rarer words, cleaner clusters
        - ↓ capture niche terms.
    - `ngram_range`: include bigrams (1,2) when phrases carry meaning.

Inputs
──────
- TXT file: One response per line.
- CSV file: A text column specified by `--text-col`.

Outputs
───────
CSV with columns: `id`, `text`, `topic`, `prob`, `label`

Usage
─────
python create_bertopics_model.py --input responses.txt --output results.csv

Dependencies
────────────
- pandas >= 2.0
- scikit-learn >= 1.3
- sentence-transformers
- umap-learn
- hdbscan
- bertopic

Limitations
───────────
- Tested on Python 3.11 (current build for BERTopics)
- UTF-8 encoding assumed
- Requires GPU or high-memory CPU for large datasets
"""

__author__  = "Aidan Calderhead"
__created__ = "2025-10-09"
__version__ = "1.1.0"
__license__ = "MIT"

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import logging
import os
import re
import sys
import tempfile
import unicodedata
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer, 
    ENGLISH_STOP_WORDS
)
import umap     # type: ignore
import hdbscan  # type: ignore
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Text I/O Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_texts(path: str, text_col: Optional[str] = None) -> pd.DataFrame:
    """
    Load text responses from a TXT or CSV file.

    Parameters
    ──────────
    path : str
        Path to input file.
    text_col : str, optional
        Column name containing text when loading a CSV file.

    Returns
    ───────
    pandas.DataFrame
        DataFrame with columns:
        - `id`   : zero-padded string index
        - `text` : text response
    """
    logger.info("Loading texts from %s", path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        if not text_col:
            raise ValueError(
                "CSV input requires --text-col to specify a text column."
            )
        df = pd.read_csv(
            path, 
            dtype           = str, 
            encoding        = "utf-8", 
            keep_default_na = False
        )
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in CSV.")
        texts = df[text_col].fillna("").astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    ids = [f"{i:05d}" for i in range(len(texts))]
    logger.info("Loaded %d responses", len(ids))
    return pd.DataFrame({"id": ids, "text": texts})


def save_dataframe_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save results to CSV.

    Parameters
    ──────────
    df : pandas.DataFrame
        DataFrame to save.
    path : str
        Destination path for output CSV.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(
        path, 
        index    = False, 
        encoding = "utf-8", 
        quoting  = csv.QUOTE_NONNUMERIC
    )
    logger.info("Saved results to %s", path)

# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def basic_clean(text: Optional[str]) -> str:
    """
    Clean and normalize text input for modeling.

    Steps
    ─────
    - Converts None → ""
    - Normalizes Unicode (NFKC form)
    - Removes invisible/zero-width characters
    - Collapses whitespace

    Parameters
    ──────────
    text : str or None
        Raw input text.

    Returns
    ───────
    str
        Cleaned and normalized text string.
    """
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────────────────────────────────────
# BERTopic Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_topic_model(
    ngram_range:     Tuple[int, int] = (1, 2),
    stop_words:      Union[str, List[str], None] = "english",
    min_df:          int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
    min_topic_size:  int = 10,
    min_samples:     int = 5,
    n_neighbors:     int = 15,
    n_components:    int = 5,
    random_state:    Optional[int] = None,
    verbose:         bool = False,
) -> BERTopic:
    """
    Construct a BERTopic model with modular tuning controls.

    Parameters
    ──────────
    ngram_range : tuple of (int, int), default=(1, 2)
        Range of n-grams for tokenization.
        Increase upper bound to include bi-grams or tri-grams 
        when phrases add value.
    stop_words : list or str, default='english'
        Stopword set for the vectorizer. Provide custom list 
        to control filtering.
    min_df : int, default=5
        Minimum number of documents a term must appear in to be kept.
        - ↑ reduces noise
        - ↓ preserves niche terms
    embedding_model : str, default='all-MiniLM-L6-v2'
        SentenceTransformer model to use for semantic embeddings.
        Larger models (e.g., "all-mpnet-base-v2") improve coherence 
        at higher cost.
    min_topic_size : int, default=10
        Minimum cluster size for HDBSCAN.
        - ↑ broader topics
        - ↓ finer-grained distinctions
    min_samples : int, default=5
        Sensitivity of HDBSCAN clustering.
        - ↑ stricter clusters (more outliers)
        - ↓ looser clustering
    n_neighbors : int, default=15
        Controls local vs global structure in UMAP.
        - ↑ smoother manifold
        - ↓ preserves fine-grained differences
    n_components : int, default=5
        Dimensionality of UMAP projection.
        - ↑ captures more variation (slower)
        - ↓ simpler representation
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, prints BERTopic progress messages.

    Returns
    ───────
    BERTopic
        Configured BERTopic model instance.
    """
    vectorizer = CountVectorizer(
        ngram_range = ngram_range,
        stop_words  = stop_words,
        min_df      = min_df,
    )

    embedder = SentenceTransformer(embedding_model)
    umap_model = umap.UMAP(
        n_neighbors  = n_neighbors,
        n_components = n_components,
        random_state = random_state,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size = min_topic_size,
        min_samples      = min_samples,
        prediction_data  = True,
    )

    model = BERTopic(
        vectorizer_model = vectorizer,
        embedding_model  = embedder,
        umap_model       = umap_model,
        hdbscan_model    = hdbscan_model,
        calculate_probabilities = True,
        verbose          = verbose,
    )

    return model

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_path:       str,
    output_path:      Optional[str] = None,
    text_col:         Optional[str] = None,
    ngram_range:      Tuple[int, int] = (1, 2),
    min_df:           int = 5,
    embedding_model:  str = "all-MiniLM-L6-v2",
    min_topic_size:   int = 10,
    min_samples:      int = 5,
    n_neighbors:      int = 15,
    n_components:     int = 5,
    reduce_topics_to: Optional[int] = None,
    n_topic_words:    int = 8,
    random_state:     Optional[int] = None,
    extra_stopwords:  Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Execute the complete BERTopic workflow: load, clean, model, label, save.

    Parameters
    ──────────
    input_path : str
        Path to input file (.txt or .csv).
    output_path : str, optional
        Destination for output CSV (default: system temp directory).
    text_col : str, optional
        Column name if CSV input provided.
    ngram_range : tuple of (int, int), default=(1, 2)
        Range of n-grams to include.
    min_df : int, default=5
        Minimum document frequency for token retention.
    embedding_model : str, default='all-MiniLM-L6-v2'
        Pretrained SentenceTransformer model for embedding.
    min_topic_size : int, default=10
        Minimum cluster size for topic formation.
    min_samples : int, default=5
        Minimum points per cluster in HDBSCAN.
    n_neighbors : int, default=15
        Neighborhood size for UMAP manifold learning.
    n_components : int, default=5
        Output dimensionality of UMAP projection.
    reduce_topics_to : int, optional
        Reduce the number of topics after fitting.
    n_topic_words : int, default=8
        Number of representative words for labeling topics.
    random_state : int, optional
        Random seed.
    extra_stopwords : list of str, optional
        User-supplied additional stopwords.

    Returns
    ───────
    pandas.DataFrame
        DataFrame with topic assignments, probabilities, and labels.
    """
    df = load_texts(input_path, text_col)
    df["text"] = df["text"].apply(basic_clean)

    stop_words = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stop_words.update(extra_stopwords)

    model = build_topic_model(
        ngram_range=ngram_range,
        stop_words=list(stop_words),
        min_df=min_df,
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        min_samples=min_samples,
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
        verbose=True,
    )

    texts = df["text"].tolist()
    topics, probs = model.fit_transform(texts)

    if reduce_topics_to:
        model = model.reduce_topics(texts, nr_topics=reduce_topics_to)
        topics, probs = model.transform(texts)

    df["topic"] = topics
    df["prob"] = [float(p.max()) if p is not None else None for p in probs]

    labels = model.generate_topic_labels(
        nr_words=n_topic_words,
        topic_prefix=True,
        separator=" | ",
    )
    topic_info = model.get_topic_info()
    label_map = dict(zip(topic_info["Topic"], labels))
    df["label"] = df["topic"].map(label_map)

    if not output_path:
        tmp_dir = tempfile.gettempdir()
        output_path = os.path.join(tmp_dir, "bertopic_results.csv")
    save_dataframe_to_csv(df, output_path)

    logger.info("Pipeline complete: %d items processed", len(df))
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CLI & Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    ───────
    argparse.Namespace
        Parsed argument object.
    """
    parser = argparse.ArgumentParser(
        description = "Generalized BERTopic modeling pipeline"
    )
    parser.add_argument(
        "--input", 
        "-i", 
        required = True, 
        help     = "Path to input file (.txt or .csv)"
    )
    parser.add_argument(
        "--output", 
        "-o", 
        help = "Path to save output CSV"
    )
    parser.add_argument(
        "--text-col", 
        "-c", 
        help = "Column name for CSV input"
    )
    parser.add_argument(
        "--embedding-model", 
        default = "all-MiniLM-L6-v2", 
        help    = "SentenceTransformer model name"
    )
    parser.add_argument(
        "--reduce-topics-to", 
        type    = int, 
        default = None, 
        help    = "Reduce topic count post-fit"
    )
    parser.add_argument(
        "--n-topic-words", 
        type    = int, 
        default = 8, 
        help    = "Number of words per topic label"
    )
    parser.add_argument(
        "--random-state", 
        type    = int, 
        default = None, 
        help    = "Random seed for reproducibility"
    )
    return parser.parse_args(argv)

# ─────────────────────────────────────────────────────────────────────────────
# Main Function
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    run_pipeline(
        input_path       = args.input,
        output_path      = args.output,
        text_col         = args.text_col,
        embedding_model  = args.embedding_model,
        reduce_topics_to = args.reduce_topics_to,
        n_topic_words    = args.n_topic_words,
        random_state     = args.random_state,
    )


if __name__ == "__main__":
    main()
