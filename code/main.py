import os
from pathlib import Path
from data_processing import load_texts_from_file
from utils import visualise_aggregated_report, compile_aggregated_report_batched, save_human_readable_report
from utils import corpus_level_balance_filter, generate_ratio_summary_parallel, clear_directory
from utils import filter_existing_knowledgebases, visualise_gender_ratio_distribution
from utils import create_balanced_corpus_parallel, save_exclusion_report
from analysis import process_single_text
from data_processing import save_knowledgebase, load_all_knowledgebases
import spacy
from concurrent.futures import ProcessPoolExecutor
import pickle
import json

def initialise_spacy_pipeline():
    """
    Initialise and return a configured SpaCy NLP pipeline for German text.

    Loads the 'de_core_news_lg' SpaCy model and adds the following components:
    - merge_entities: Merges multi-token named entities into single tokens.
    - coreferee: Adds co-reference resolution functionality.

    Returns:
        spacy.language.Language: The initialised SpaCy NLP pipeline.
    """
    print("[INFO] Initialising SpaCy pipeline...")
    nlp = spacy.load("de_core_news_lg")
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("coreferee")
    print("[INFO] SpaCy pipeline initialised.")
    return nlp


def process_single_file_multiprocessing(file_path, temp_dir, max_articles=None):
    """
    Process a single JSON file of articles using SpaCy and save one knowledgebase per article.

    This function performs full text processing for each article, including:
    - Named actor and pronoun extraction
    - Gender coding and nomination detection
    - Sentiment analysis
    - Coreference resolution (via SpaCy + Coreferee)
    - Detection of gender-neutral and generic masculine language
    - PMI-based key term analysis

    Each article results in:
    - A pickled actor-level knowledgebase (`knowledgebase_<article_id>.pkl`)
    - A pickled metadata file for text-level features (`meta_data_<article_id>.pkl`)

    Args:
        file_path (str): Path to the JSON article file.
        temp_dir (Path): Directory to save output `.pkl` files.
        max_articles (int, optional): Maximum number of articles to process from the file.

    Returns:
        int: Number of successfully processed articles.

    Notes:
        - This function is multiprocessing-compatible.
        - The SpaCy pipeline is reinitialised within each process.
    """

    try:
        print(f"[INFO] Processing file: {file_path}")

        nlp = initialise_spacy_pipeline()
        stopwords = nlp.Defaults.stop_words
                
        print(f"[INFO] SpaCy pipeline initialised for {file_path}")

        raw_texts = load_texts_from_file(file_path, limit=max_articles)
        article_count = 0

        file_stem = Path(file_path).stem  # "taz_1986-04"
        year, month = file_stem.split("_")[1].split("-")
        
        for article_id, text in raw_texts.items():

           try:
                try:
                    doc = nlp(text)
                except Exception as e:
                    print(f"[ERROR] NLP parsing failed for Article ID {article_id}: {e}")
                    continue

                try:
                    knowledgebase, uses_gnl, generic_masc = process_single_text(doc, stopwords)
                except Exception as e:
                    print(f"[ERROR] Text processing failed for Article ID {article_id}: {e}")
                    continue

                if knowledgebase.empty:
                    continue
                    
                # save actor based knowledgebase
                knowledgebase["article_id"] = article_id
                save_knowledgebase(knowledgebase, temp_dir, article_id)

                # save meta_data for whole text
                meta_data = {
                    "article_id": article_id,
                    "uses_gender_neutral_language": uses_gnl,
                    "generic_masculine": generic_masc,
                    "year": year,
                    "month": month
                }
                with open(temp_dir / f"meta_data_{article_id}.pkl", "wb") as file:
                    pickle.dump(meta_data, file)

                article_count += 1
                    
           except Exception as e:
                print(f"[ERROR] Failed to process Article ID {article_id}: {e}")

        return article_count
                
    except Exception as e:
        print(f"[ERROR] Error processing file {file_path}: {e}")
        return 0


def process_directory(directory, temp_dir, max_articles=None, max_workers=None):
    """
    Process all JSON article files in a directory using multiprocessing.

    This function:
    - Iterates over all `.json` files in the specified input directory.
    - Uses multiple worker processes to run `process_single_file_multiprocessing` on each file.
    - Performs actor-level extraction, gender annotation, sentiment analysis, and coreference resolution.
    - Stores the results as one `.pkl` file per article:
        * `knowledgebase_<article_id>.pkl` (actor-level data)
        * `meta_data_<article_id>.pkl` (document-level flags)

    Args:
        directory (str): Path to the folder containing raw article `.json` files.
        temp_dir (Path): Temporary directory to save the output `.pkl` files.
        max_articles (int, optional): Limit on number of articles processed per file. Defaults to None (all articles).
        max_workers (int, optional): Number of parallel worker processes. Defaults to available CPU cores minus 2.

    Returns:
        int: Total number of articles successfully processed across all files.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    json_files = [os.path.join(directory, file_name) for file_name in os.listdir(directory) if file_name.endswith(".json")]

    total_texts = 0
    max_workers = max_workers or max(1, os.cpu_count() - 2)

    print(f"[INFO] Using {max_workers} parallel workers.")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_file_multiprocessing, file_path, temp_dir, max_articles)
            for file_path in json_files
        ]

        for future in futures:
            try:
                count = future.result()
                total_texts += count
            except Exception as e:
                print(f"[ERROR] Error during processing: {e}")

    return total_texts

    

def analyse_and_visualise_year(directory, temp_dir, results_dir, year, max_articles=None, max_workers=None):
    """
    Perform a full-year analysis and visualisation pipeline on newspaper texts using batched processing.

    This function processes all article JSON files in the specified directory for a given year.
    It creates batched actor-level knowledgebases, compiles an aggregated report,
    saves a human-readable summary, and generates corresponding visualisations.

    Steps performed:
    1. Processes all articles in `directory` using multiprocessing and stores batched
       actor-level knowledgebases and flags in `temp_dir`.
    2. Loads batched knowledgebases and flag files, compiles an aggregated report.
    3. Saves a human-readable `.txt` report summarising the year’s metrics.
    4. Generates and stores boxplots and barplots illustrating key gender metrics.

    Args:
        directory (Path or str): Path to the folder containing article `.json` files for the given year.
        temp_dir (Path): Directory for storing per-article `.pkl` output files.
        results_dir (Path): Directory where the report and visualisations will be saved.
        year (int): The year being analysed (used in filenames and titles).
        max_articles (int, optional): Maximum number of articles to process per file (default: None = all).
        max_workers (int, optional): Number of parallel worker processes (default: None).

    Outputs:
        - `aggregated_report_<year>.txt`: Human-readable textual report of average and median gender metrics.
        - Various PNG boxplots and barplots visualising gender, sentiment, and representation metrics.

    Notes:
        - Expects batched `.pkl` files: `knowledgebase_batch_*.pkl` in `temp_dir`.
        - Does not resave knowledgebases — those remain in `temp_dir`.
    """
    print(f"[INFO] Starting analysis for {year}...")
    total_texts = process_directory(directory, temp_dir, max_articles, max_workers=max_workers)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load all knowledgebases
    print(f"[INFO] Loading individual knowledgebases from {temp_dir}...")
    all_knowledgebases = load_all_knowledgebases(temp_dir)

    if all_knowledgebases:
        print(f"[INFO] Compiling aggregated report for {year} from batched knowledgebases...")
        aggregated_report, individual_values = compile_aggregated_report_batched(temp_dir)

        print(f"[INFO] Aggregated report for {year} generated.")

        # Save human-readable report
        save_human_readable_report(aggregated_report, results_dir, year, total_texts)

        # Generate visualisations
        print(f"[INFO] Saving visualisations for {year} to {results_dir}...")
        visualise_aggregated_report(individual_values, results_dir)
        print(f"[INFO] Visualisations saved for {year}.")

    else:
        print(f"[INFO] No valid knowledgebase batches found for {year}. Skipping report.")


def main():
    """
    Entry point for running the gender discrimination analysis pipeline.

    This function prompts the user to select between two operational modes:

    Mode 1 - Standard Analysis:
        - Iterates over year-based subdirectories in the `corpus` folder.
        - For each year, it processes newspaper articles, extracts actor-level data,
          and computes gender-related metrics.
        - Generates and saves:
            * Aggregated per-year knowledgebases (.pkl),
            * Human-readable text reports,
            * Visualisations (boxplots and barplots).

    Mode 2 - Balanced Corpus Recommendation:
        - Processes the entire corpus and generates actor-level knowledgebases into `tmp_2/`.
         - Applies a multi-stage filtering and balancing process:
            Step 1: Generate actor-level knowledgebases for all articles.
            Step 2: Filter texts based on gender asymmetries (e.g. sentiment gap, naming imbalance).
            Step 3: Perform corpus-level balancing to remove remaining distortions in gender ratios.
            Step 4: Create a balanced output corpus (`balanced_corpus/`) and a full exclusion report.
        - Produces and saves three gender ratio histograms to visualise filtering effects:
            * Before exclusion
            * After discriminatory article removal
            * After corpus-level balancing
        - Outputs:
            * `excluded_index.json`: Articles excluded with month-level structure.
            * `exclusion_report.txt`: Summary of exclusion steps and counts.
            * Visual diagnostics:
                - `gender_ratio_before_exclusion.png`
                - `gender_ratio_after_filtering.png`
                - `gender_ratio_after_balancing.png`

    Input and Output Structure:
        - Input: `corpus/YYYY/*.json` (with one subdirectory per year)
        - Intermediate:
            * `tmp_1/YYYY/` for Mode 1
            * `tmp_2/` for Mode 2
        - Output:
            * `results_1/YYYY/` (Mode 1 visualisations + reports)
            * `results_2/` (Mode 2 diagnostics + summary files)
            * `balanced_corpus/` (Mode 2 filtered article set)

    Notes:
        - Multiprocessing is used to speed up per-file processing.
        - Assumes all year folders in `corpus/` are named using four-digit numbers (e.g. "1984").
        - Prompts the user for filtering thresholds and exclusion rules in Mode 2.

    Raises:
        - Prints errors if `corpus` folder is missing or user input is invalid.
    """
    CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
    EXCLUSION_FILE = Path("excluded_index.json")
    BALANCED_CORPUS_DIR = Path(__file__).resolve().parents[1] / "balanced_corpus"
    MAX_WORKERS = 5

    if not CORPUS_DIR.exists():
        print(f"[ERROR] Corpus directory not found: {CORPUS_DIR}")
        return

    print("Choose mode:\n"
          "1 - Analyse corpus per year\n"
          "2 - Generate balanced corpus recommendation (exclude discriminatory texts)")
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice in {"1", "2"}:
        mode = choice
    else:
        print("[ERROR] Invalid choice. Please enter 1 or 2.")
        return

    directories = {year.name: str(year) for year in CORPUS_DIR.iterdir() if year.is_dir() and year.name.isdigit()}

    if not directories:
        print("[WARNING] No year directories found in the corpus.")
        return

    if mode == "1":
        for year, directory in sorted(directories.items()):
            print(f"\n[INFO] Starting processing for year {year}...\n")
            year_temp_dir = Path("tmp_1") / year
            year_results_dir = Path("results_1") / year

            analyse_and_visualise_year(
                directory=directory,
                temp_dir=year_temp_dir,
                results_dir=year_results_dir,
                year=year,
                max_articles=None,
                max_workers=MAX_WORKERS
            )

    elif mode == "2":
        if EXCLUSION_FILE.exists():
            EXCLUSION_FILE.unlink()

        tmp_2 = Path("tmp_2")
        results_dir = Path("results_2")

        tmp_2.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        step1_count = 0
        step2_excluded_count = 0
        step2_flag_counts = {"sentiment_gap": 0, "subject_gap": 0, "quote_gap": 0, "naming_gap": 0}
        step3_excluded_count = 0

        # Step 1: One full pass to generate knowledgebases
        print("[STEP 1] Creating knowledgebases for the full corpus...")
        for year, directory in sorted(directories.items()):
            print(f"[INFO] Processing year {year}...")
            step1_count += process_directory(
                directory=directory,
                temp_dir=tmp_2,
                max_articles=None,
                max_workers=MAX_WORKERS
            )

        generate_ratio_summary_parallel(temp_dir=tmp_2, summary_dir=tmp_2 / "summaries")
        visualise_gender_ratio_distribution(
            summary_dir=tmp_2 / "summaries",
            output_file=results_dir / "gender_ratio_before_exclusion.png"
        )

        # Step 2: Text-level filtering
        print("\n[STEP 2] Configure text-level exclusion")
        try:
            min_flags = int(input("How many flags should be required to exclude a discriminatory text? (e.g. 2): ").strip())
            sentiment_thresh = float(input("Set sentiment gap threshold (default = 0.3): ").strip() or 0.3)
            role_thresh = float(input("Set subject/object role gap threshold (default = 0.5): ").strip() or 0.5)
            quote_thresh = float(input("Set quote gap threshold (default = 0.5): ").strip() or 0.5)
            naming_thresh = float(input("Set naming/pronoun gap threshold (default = 0.5): ").strip() or 0.5)
        except ValueError:
            print("[ERROR] Invalid input. Using default values.")
            min_flags = 2
            sentiment_thresh = 0.3
            role_thresh = 0.5
            quote_thresh = 0.5
            naming_thresh = 0.5

        print("[INFO] Applying discriminatory text filtering...")

        excluded_index = filter_existing_knowledgebases(
            input_dir=tmp_2,
            output_dir=tmp_2,
            exclusion_file=EXCLUSION_FILE,
            sentiment_threshold=sentiment_thresh,
            role_threshold=role_thresh,
            quote_threshold=quote_thresh,
            naming_threshold=naming_thresh,
            min_flags=min_flags
        )

        step2_excluded_count = sum(len(ids) for ids in excluded_index.values())
        for ids in excluded_index.values():
            for article_id in ids:
                flag_path = tmp_2 / f"flags_{article_id}.pkl"
                if flag_path.exists():
                    with open(flag_path, "rb") as f:
                        flags = pickle.load(f)
                    for k in step2_flag_counts:
                        if flags.get(k):
                            step2_flag_counts[k] += 1

        clear_directory(tmp_2 / "summaries")
        generate_ratio_summary_parallel(temp_dir=tmp_2, summary_dir=tmp_2 / "summaries")
        visualise_gender_ratio_distribution(
            summary_dir=tmp_2 / "summaries",
            output_file=results_dir / "gender_ratio_after_filtering.png"
        )

        # Step 3: Corpus-level balancing
        print("\n[STEP 3] Configure corpus-level balancing thresholds")
        try:
            equilibrium_lo = float(input("Set equilibrium lower bound (default = 0.75): ").strip() or 0.75)
            equilibrium_hi = float(input("Set equilibrium upper bound (default = 1.25): ").strip() or 1.25)
        except ValueError:
            print("[ERROR] Invalid input. Using default values.")
            equilibrium_lo = 0.75
            equilibrium_hi = 1.25

        print("[INFO] Running corpus-level balancing...")

        corpus_level_balance_filter(
            tmp_dir=tmp_2,
            exclusion_file_path=EXCLUSION_FILE,
            equilibrium_lower=equilibrium_lo,
            equilibrium_upper=equilibrium_hi
        )

        clear_directory(tmp_2 / "summaries")
        generate_ratio_summary_parallel(temp_dir=tmp_2, summary_dir=tmp_2 / "summaries")
        visualise_gender_ratio_distribution(
            summary_dir=tmp_2 / "summaries",
            output_file=results_dir / "gender_ratio_after_balancing.png"
        )

        print(f"[INFO] Final excluded article IDs saved to: {EXCLUSION_FILE.resolve()}")

        with open(EXCLUSION_FILE, "r", encoding="utf-8") as f:
            exclusion_index_final = json.load(f)
        step3_excluded_count = sum(len(ids) for ids in exclusion_index_final.values())

        # Step 4: Create structured exclusion index and new corpus
        print("\n[STEP 4] Creating new balanced corpus...")
        create_balanced_corpus_parallel(CORPUS_DIR, EXCLUSION_FILE, BALANCED_CORPUS_DIR)

        save_exclusion_report(
            step1_count,
            step2_excluded_count,
            step2_flag_counts,
            step3_excluded_count,
            output_file=results_dir / "exclusion_report.txt"
        )

    else:
        print("[ERROR] Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
