import os
import json
from pathlib import Path
import pickle
import pandas as pd

    
def load_texts_from_file(file_path, limit=None):
    """
    Load article texts from a JSON file and return them as a dictionary.

    This function reads a JSON file where each entry is expected to contain a nested
    text field under 'text' -> 'text'. It filters out empty or invalid entries and optionally
    limits the number of articles loaded.

    Args:
        file_path (str): Path to the JSON file containing article data.
        limit (int, optional): Maximum number of articles to load. If None, all articles are loaded.

    Returns:
        dict: Dictionary mapping article IDs (str) to raw text content (str).
              Returns an empty dictionary in case of an error.
    """
    try:
        # Ensure the file exists and is accessible
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Create a dictionary for raw texts
        raw_texts = {}
        count = 0
        for article_id, article_data in data.items():
            # Stop if the limit is reached
            if limit is not None and count >= limit:
                break

            # Extract and validate the text
            raw_text = article_data.get("text", {}).get("text", "").strip()
            if raw_text:
                raw_texts[article_id] = raw_text
                count += 1
            else:
                print(f"[DEBUG] Skipping empty or invalid text for Article ID: {article_id}")

        print(f"[INFO] Loaded {len(raw_texts)} articles from {file_path}.")
        return raw_texts

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON file {file_path}: {e}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while loading texts: {e}")

    # Return an empty dictionary if an error occurs
    return {}


def load_all_knowledgebases(temp_dir):
    """
    Load all non-empty knowledgebase DataFrames from the specified directory.

    This function searches for all files matching the pattern 'knowledgebase_*.pkl' in the
    provided directory. Each file is expected to be a pickled pandas DataFrame.
    Only non-empty DataFrames are appended to the resulting list.

    Args:
        temp_dir (str or Path): Path to the directory containing knowledgebase files.

    Returns:
        list[pd.DataFrame]: List of loaded and non-empty knowledgebase DataFrames.
    """
    knowledgebases = []
    for file_path in Path(temp_dir).glob("knowledgebase_*.pkl"):
        try:
            print(f"[INFO] Loading {file_path.name}...")
            knowledgebase = load_knowledgebase(file_path)
            if not knowledgebase.empty:
                knowledgebases.append(knowledgebase)
                print(f"[INFO] Knowledgebase loaded with {len(knowledgebase)} entries.")
            else:
                print(f"[WARNING] {file_path.name} is empty, skipping.")
        except Exception as e:
            print(f"[ERROR] Failed to load knowledgebase from {file_path}: {e}")
    return knowledgebases


def load_knowledgebase(file_path):
    """
    Load a knowledgebase from a pickle (.pkl) file.

    This function attempts to open and deserialize a pickled pandas DataFrame from
    the specified file path. If loading fails, an empty DataFrame is returned.

    Args:
        file_path (str or Path): Path to the pickle file containing the knowledgebase.

    Returns:
        pd.DataFrame: The loaded knowledgebase DataFrame, or an empty DataFrame if an error occurs.
    """
    try:
        with open(file_path, "rb") as f:
            knowledgebase = pickle.load(f)
        print(f"[INFO] Knowledgebase loaded from {file_path}.")
        return knowledgebase
    except Exception as e:
        print(f"[ERROR] Failed to load knowledgebase from {file_path}: {e}")
        return pd.DataFrame()


def save_knowledgebase(knowledgebase, tmp_dir, article_id):
    """
    Save a single article's knowledgebase as a pickle (.pkl) file.

    Converts any token-like objects in specific columns to plain text before saving.
    Ensures the output directory exists and skips saving if the knowledgebase is empty.

    Args:
        knowledgebase (pd.DataFrame): The DataFrame representing actor-predication data for a single article.
        tmp_dir (str or Path): The directory where the pickle file should be saved.
        article_id (str): Identifier of the article, used to name the output file.

    Notes:
        Columns 'nomination' and 'pronoun' are converted to lists of strings
        before saving to ensure compatibility with pickle serialization.

    Returns:
        None
    """
    if knowledgebase.empty:
        print(f"[DEBUG] Knowledgebase {article_id} is empty, skipping.")
        return

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_path = tmp_dir / f"knowledgebase_{article_id}.pkl"

    try:
        knowledgebase = knowledgebase.copy()
        for column in ["nomination", "pronoun"]:
            if column in knowledgebase.columns:
                knowledgebase[column] = knowledgebase[column].apply(
                    lambda tokens: [token.text if hasattr(token, "text") else token for token in tokens]
                )

        with open(file_path, "wb") as file:
            pickle.dump(knowledgebase, file)
        print(f"[INFO] Knowledgebase saved to {file_path}.")
    except Exception as e:
        print(f"[ERROR] Failed to save knowledgebase for Article ID {article_id}: {e}")


def _filter_file_from_index(file_path, excluded_ids, output_base):
    """
        Filters a single JSON file by removing articles listed in excluded_ids.

        Assumes input is a list of article dicts, each with an 'id' field.
        Writes filtered list to a file in the same relative structure under output_base.

        Args:
            file_path (Path): Path to the original taz_YYYY-MM.json file.
            excluded_ids (set): Set of article IDs (as strings) to exclude.
            output_base (Path): Root directory where the balanced corpus will be saved.

        Returns:
            tuple: (original_count, filtered_count)
        """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        filtered = [article for article in data if str(article.get("id")) not in excluded_ids]

        if filtered:
            target_dir = output_base / file_path.parent.name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / file_path.name

            with open(target_file, "w", encoding="utf-8") as out:
                json.dump(filtered, out, ensure_ascii=False)

        return len(data), len(filtered)

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return 0, 0