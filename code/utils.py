import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processing import load_all_knowledgebases,_filter_file_from_index
from collections import Counter, defaultdict
import pickle
from pathlib import Path
import numbers
from concurrent.futures import ProcessPoolExecutor
import json
from functools import partial
import math
import matplotlib as mpl



def save_human_readable_report(aggregated_report, results_dir, year, total_texts):
    """
    Save a comprehensive, human-readable summary of yearly gender representation metrics.

    This function generates a plain-text report (`aggregated_report_<year>.txt`) that summarises:
    - Global corpus-level statistics (e.g. total texts, number of actor-containing texts)
    - Aggregated totals for key gender indicators by pronoun group (she/her vs he/him)
    - Relative proportions (e.g. subject vs object roles, quote types)
    - Per-text statistics: mean, median, and standard deviation
    - Top PMI-associated terms (adjectives, nouns, verbs), both overall and by pronoun group

    Output Format:
    - Aligned table of metrics showing counts and proportions by group
    - Per-text statistics for comparative analysis across documents
    - PMI keyword tables for each part-of-speech, grouped by she/her, he/him, and overall

    Args:
        aggregated_report (dict): Summary statistics compiled across all articles for the year.
        results_dir (Path): Output directory where the report should be saved.
        year (int): The year associated with the dataset (used in the report filename).
        total_texts (int): Total number of articles processed for the given year.

    Output:
        A plain-text report file named `aggregated_report_<year>.txt` in the `results_dir`.

    Notes:
        - This report complements the visualisations and is suitable for qualitative inspection.
        - All numeric outputs are formatted for readability and grouped by metric type.
    """
    report_path = results_dir / f"aggregated_report_{year}.txt"

    # Helper functions
    def percent(part, total):
        return round(100 * part / total, 1) if total else 0.0

    def write_total_line(label, val_she, val_he):
        return f"{label:35}{val_she:12}{val_he:12}{val_she + val_he:12}\n"

    def write_percentage_line(label, val_she, val_he, total):
        return f"{label:35}{percent(val_she, total):12.1f}{percent(val_he, total):12.1f}{'':12}\n"

    def write_mean_line(label, val_she, val_he):
        values = [v for v in [val_she, val_he] if v is not None]
        overall = round(sum(values) / len(values), 2) if values else 0.0
        return f"{label:35}{val_she:12.2f}{val_he:12.2f}{overall:12.2f}\n"

    try:
        with open(report_path, "w", encoding="utf-8") as report:

            report.write("\n")
            report.write(f"Report for the year {year}\n")
            report.write("=" * 75 + "\n\n")

            # AGGREGATED TOTALS
            report.write("AGGREGATED TOTALS (all texts)\n")
            report.write(f"{'Total Texts:':35}{total_texts:>6}\n")
            report.write(f"{'Texts with Actors:':35}{aggregated_report['texts_with_actors']:>6}\n")
            report.write(
                f"{'Uses Gender Neutral Language (Docs):':35}{aggregated_report.get('uses_gender_neutral_language', 0):>5}\n")
            report.write(
                f"{'Generic Masculine Usage (Docs):':35}{aggregated_report.get('generic_masculine', 0):>6}\n\n")

            report.write(f"{'Metric':35}{'she/her':>12}{'he/him':>12}{'overall':>12}\n")
            report.write("-" * 75 + "\n")

            metrics_raw = {
                "Pronoun Distribution:": (
                    aggregated_report.get("pronoun_distribution_she_her", 0),
                    aggregated_report.get("pronoun_distribution_he_him", 0)
                ),
                "Mentions by Pronoun:": (
                    aggregated_report.get("mentions_pronoun_distribution_she_her", 0),
                    aggregated_report.get("mentions_pronoun_distribution_he_him", 0)
                ),
                "Named Mentions:": (
                    aggregated_report.get("name_count_she_her", 0),
                    aggregated_report.get("name_count_he_him", 0)
                ),
                "Pronoun Mentions:": (
                    aggregated_report.get("pronoun_count_she_her", 0),
                    aggregated_report.get("pronoun_count_he_him", 0)
                ),
                "Subject Roles:": (
                    aggregated_report.get("subject_count_she_her", 0),
                    aggregated_report.get("subject_count_he_him", 0)
                ),
                "Object Roles:": (
                    aggregated_report.get("object_count_she_her", 0),
                    aggregated_report.get("object_count_he_him", 0)
                ),
                "Direct Quotes:": (
                    aggregated_report.get("direct_quote_count_she_her", 0),
                    aggregated_report.get("direct_quote_count_he_him", 0)
                ),
                "Indirect Quotes:": (
                    aggregated_report.get("indirect_quote_count_she_her", 0),
                    aggregated_report.get("indirect_quote_count_he_him", 0)
                ),
                "Feminine-coded Words:": (
                    aggregated_report.get("feminine_coded_words_pronoun_distribution_she_her", 0),
                    aggregated_report.get("feminine_coded_words_pronoun_distribution_he_him", 0)
                ),
                "Masculine-coded Words:": (
                    aggregated_report.get("masculine_coded_words_pronoun_distribution_she_her", 0),
                    aggregated_report.get("masculine_coded_words_pronoun_distribution_he_him", 0)
                ),
                "Sentiment:": (
                    aggregated_report.get("sentiment_by_pronoun", {}).get("she_her", 0.0),
                    aggregated_report.get("sentiment_by_pronoun", {}).get("he_him", 0.0)
                )
            }

            for label, (val_she, val_he) in metrics_raw.items():
                if "Sentiment" in label:
                    report.write(write_mean_line(label, val_she, val_he))
                else:
                    report.write(write_total_line(label, val_she, val_he))

            name_she = metrics_raw["Named Mentions:"][0]
            name_he = metrics_raw["Named Mentions:"][1]
            pronoun_she = metrics_raw["Pronoun Mentions:"][0]
            pronoun_he = metrics_raw["Pronoun Mentions:"][1]
            total_mentions = name_she + name_he + pronoun_she + pronoun_he
            subj_she = metrics_raw["Subject Roles:"][0]
            subj_he = metrics_raw["Subject Roles:"][1]
            obj_she = metrics_raw["Object Roles:"][0]
            obj_he = metrics_raw["Object Roles:"][1]
            total_roles = subj_she + subj_he + obj_she + obj_he
            direct_she = metrics_raw["Direct Quotes:"][0]
            direct_he = metrics_raw["Direct Quotes:"][1]
            indirect_she = metrics_raw["Indirect Quotes:"][0]
            indirect_he = metrics_raw["Indirect Quotes:"][1]
            total_quotes = direct_she + direct_he + indirect_she + indirect_he

            report.write(write_percentage_line("Named Mentions (% of all mentions):", name_she, name_he, total_mentions))
            report.write(write_percentage_line("Pronoun Mentions (% of all mentions):", pronoun_she, pronoun_he, total_mentions))
            report.write(write_percentage_line("Subject Roles (% of known roles):", subj_she, subj_he, total_roles))
            report.write(write_percentage_line("Object Roles (% of known roles):", obj_she, obj_he, total_roles))
            report.write(write_percentage_line("Direct Quotes (% of quotes):", direct_she, direct_he, total_quotes))
            report.write(write_percentage_line("Indirect Quotes (% of quotes):", indirect_she, indirect_he, total_quotes))
            report.write("\n")

            # PER TEXT METRICS
            mean = aggregated_report.get("mean_metrics", {})
            median = aggregated_report.get("median_metrics", {})
            std = aggregated_report.get("std_metrics", {})

            report.write("STATISTICS (per text)\n")
            report.write("-" * 75 + "\n")
            report.write(f"{'Metric':45}{'Mean':>10}{'Median':>10}{'Std Dev':>10}\n")
            report.write("-" * 75 + "\n")

            for key in mean.keys():
                label = key.replace("feminine_coded_words_pronoun_distribution", "Feminine Coded Words (by pronoun)") \
                    .replace("masculine_coded_words_pronoun_distribution", "Masculine Coded Words (by pronoun)") \
                    .replace("mentions_pronoun_distribution", "Mentions (by pronoun)") \
                    .replace("pronoun_distribution", "Pronouns (resolved)") \
                    .replace("sentiment_by_pronoun", "Sentiment (by pronoun)") \
                    .replace("mean_sentiment_all", "Mean Sentiment (all)") \
                    .replace("uses_gender_neutral_language", "Uses Gender-Neutral Language") \
                    .replace("generic_masculine", "Generic Masculine") \
                    .replace("name_count", "Named Mentions (sum over actors)") \
                    .replace("pronoun_count", "Pronoun Mentions (sum over actors)") \
                    .replace("subject_count", "Subject Roles") \
                    .replace("object_count", "Object Roles") \
                    .replace("direct_quote_count", "Direct Quotes") \
                    .replace("indirect_quote_count", "Indirect Quotes") \
                    .replace("she_her", "(she/her)") \
                    .replace("he_him", "(he/him)") \
                    .replace("percent_named_mentions", "Named Mentions (%)") \
                    .replace("percent_pronoun_mentions", "Pronoun Mentions (%)") \
                    .replace("percent_subject_roles", "Subject Roles (%)") \
                    .replace("percent_object_roles", "Object Roles (%)") \
                    .replace("percent_direct_quotes", "Direct Quotes (%)") \
                    .replace("percent_indirect_quotes", "Indirect Quotes (%)") \
                    .replace("_", " ").strip().title()

                s = std.get(key, None)
                std_str = f"{s:.2f}" if s is not None and not pd.isna(s) else "   –"
                report.write(f"{label:45}{mean.get(key, 0):10.2f}{median.get(key, 0):10.2f}{std_str:>10}\n")

            report.write("\n")

            # PMI TABLES
            write_top_pmi_table(report, "Adjectives",
                                aggregated_report["top_pmi_adj"],
                                aggregated_report["top_pmi_adj_pronoun_distribution"].get("she_her", []),
                                aggregated_report["top_pmi_adj_pronoun_distribution"].get("he_him", []))
            report.write("\n")

            write_top_pmi_table(report, "Nouns",
                                aggregated_report["top_pmi_noun"],
                                aggregated_report["top_pmi_noun_pronoun_distribution"].get("she_her", []),
                                aggregated_report["top_pmi_noun_pronoun_distribution"].get("he_him", []))
            report.write("\n")

            write_top_pmi_table(report, "Verbs",
                                aggregated_report["top_pmi_verb"],
                                aggregated_report["top_pmi_verb_pronoun_distribution"].get("she_her", []),
                                aggregated_report["top_pmi_verb_pronoun_distribution"].get("he_him", []))

        print(f"[INFO] Human-readable report saved to {report_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save human-readable report: {e}")


def write_top_pmi_table(report, label, all_words, she_words, he_words):
    """
    Write a formatted table of top PMI-ranked words to a text report.

    This function creates a side-by-side table showing the most characteristic adjectives, nouns,
    or verbs (based on pointwise mutual information) across:
    - the full dataset
    - she/her-associated entities
    - he/him-associated entities

    The output is aligned for readability and padded to ensure equal row lengths.

    Args:
        report (TextIO): An open text file or stream to write into (typically from open(..., "w")).
        label (str): A label for the table, e.g. "Adjectives", "Nouns", or "Verbs".
        all_words (list of tuple): Top PMI terms overall as (word, score) tuples.
        she_words (list of tuple): Top PMI terms for the "she/her" group.
        he_words (list of tuple): Top PMI terms for the "he/him" group.

    Output:
        Writes a plain-text table with three aligned columns into the report stream.
        Pads shorter lists with empty values to ensure consistent formatting.
    """
    report.write(f"TOP PMI {label.upper()}\n")
    report.write("-" * 95 + "\n")
    report.write(f"Most frequent {label.lower()} associated with each pronoun group.\n\n")
    report.write(f"{'Rank':<5}{'ALL':<30}{'she/her':<30}{'he/him':<30}\n")
    report.write("-" * 95 + "\n")
    max_len = max(len(all_words), len(she_words), len(he_words))
    all_words += [("", 0)] * (max_len - len(all_words))
    she_words += [("", 0)] * (max_len - len(she_words))
    he_words += [("", 0)] * (max_len - len(he_words))

    for i in range(max_len):
        all_word, all_pmi = all_words[i]
        she_word, she_pmi = she_words[i]
        he_word, he_pmi = he_words[i]

        all_display = f"{all_word} ({all_pmi:.2f})" if all_word else ""
        she_display = f"{she_word} ({she_pmi:.2f})" if she_word else ""
        he_display = f"{he_word} ({he_pmi:.2f})" if he_word else ""

        report.write(f"{i + 1:<5}{all_display:<30}{she_display:<30}{he_display:<30}\n")

    report.write("\n")


def generate_individual_report(knowledgebase, uses_gender_neutral_language=False, generic_masculine=False):
    """
    Generate a dictionary of gender-related metrics for a single article.

    This function processes a single DataFrame of extracted actor-level information to compute:
    - Total number of actors and mentions
    - Pronoun distributions and mention counts by gender group
    - Sentiment scores overall and by pronoun group (she/her, he/him)
    - Frequency of feminine- and masculine-coded language
    - Top PMI-associated adjectives, nouns, and verbs (overall and by pronoun group)
    - Grammatical and referential roles per gender group (e.g. subject/object mentions, quote types)
    - Flags for gender-neutral language and generic masculine usage (passed from metadata)

    Args:
        knowledgebase (pd.DataFrame): Actor-level data for one article (one row per actor-predication).
        uses_gender_neutral_language (bool): Whether the article contains gender-neutral forms.
        generic_masculine (bool): Whether the article uses generic masculine constructions.

    Returns:
        dict: Dictionary containing all extracted statistics. Keys include:
            - total_actors, total_mentions
            - pronoun_distribution, mentions_pronoun_distribution
            - total_feminine_coded_words, total_masculine_coded_words
            - feminine_coded_words_pronoun_distribution, masculine_coded_words_pronoun_distribution
            - mean_sentiment, sentiment_by_pronoun
            - top_pmi_adj / noun / verb (overall PMI terms)
            - top_pmi_*_pronoun_distribution (PMI by she/her and he/him groups)
            - name_count_*, pronoun_count_*, subject_count_*, object_count_*,
              direct_quote_count_*, indirect_quote_count_* (separated by pronoun group)
    """
    if knowledgebase.empty:
        return {}

    try:
        total_actors = len(knowledgebase)
        total_mentions = knowledgebase["mention_count"].sum()
        pronoun_distribution = knowledgebase["main_pronoun"].value_counts().to_dict()
        mentions_pronoun_distribution = knowledgebase.groupby("main_pronoun")["mention_count"].sum().to_dict()

        total_feminine_coded_words = knowledgebase["feminine_coded_words"].sum()
        total_masculine_coded_words = knowledgebase["masculine_coded_words"].sum()

        feminine_coded_words_pronoun_distribution = knowledgebase.groupby("main_pronoun")[
            "feminine_coded_words"].sum().to_dict()
        masculine_coded_words_pronoun_distribution = knowledgebase.groupby("main_pronoun")[
            "masculine_coded_words"].sum().to_dict()

        mean_sentiment_all = knowledgebase["sentiment"].mean()
        sentiment_by_pronoun = knowledgebase.groupby("main_pronoun")["sentiment"].mean().to_dict()

        top_pmi_adj = Counter(word for d in knowledgebase["pmi_adjective"] for word in (d or {})).most_common(10)
        top_pmi_noun = Counter(word for d in knowledgebase["pmi_noun"] for word in (d or {})).most_common(10)
        top_pmi_verb = Counter(word for d in knowledgebase["pmi_verb"] for word in (d or {})).most_common(10)

        top_pmi_adj_pronoun_distribution = {
            pronoun: Counter(word for d in group["pmi_adjective"] for word in (d or {})).most_common(10)
            for pronoun, group in knowledgebase.groupby("main_pronoun")
        }
        top_pmi_noun_pronoun_distribution = {
            pronoun: Counter(word for d in group["pmi_noun"] for word in (d or {})).most_common(10)
            for pronoun, group in knowledgebase.groupby("main_pronoun")
        }
        top_pmi_verb_pronoun_distribution = {
            pronoun: Counter(word for d in group["pmi_verb"] for word in (d or {})).most_common(10)
            for pronoun, group in knowledgebase.groupby("main_pronoun")
        }

        group_metrics = {}
        for group in ["she_her", "he_him"]:
            group_data = knowledgebase[knowledgebase["main_pronoun"] == group]

            name_count = group_data["actor_nomination"].apply(len).sum()
            pronoun_count = group_data["actor_pronouns"].apply(len).sum()
            subject_count = group_data["subject_role"].sum()
            object_count = group_data["object_role"].sum()
            direct_quote_count = group_data["direct_quotes"].sum()
            indirect_quote_count = group_data["indirect_quotes"].sum()

            # Absolute counts
            group_metrics[f"name_count_{group}"] = name_count
            group_metrics[f"pronoun_count_{group}"] = pronoun_count
            group_metrics[f"subject_count_{group}"] = subject_count
            group_metrics[f"object_count_{group}"] = object_count
            group_metrics[f"direct_quote_count_{group}"] = direct_quote_count
            group_metrics[f"indirect_quote_count_{group}"] = indirect_quote_count

        return {
            "total_actors": total_actors,
            "total_mentions": total_mentions,
            "pronoun_distribution": pronoun_distribution,
            "mentions_pronoun_distribution": mentions_pronoun_distribution,
            "uses_gender_neutral_language": uses_gender_neutral_language,
            "generic_masculine": generic_masculine,
            "total_feminine_coded_words": total_feminine_coded_words,
            "total_masculine_coded_words": total_masculine_coded_words,
            "feminine_coded_words_pronoun_distribution": feminine_coded_words_pronoun_distribution,
            "masculine_coded_words_pronoun_distribution": masculine_coded_words_pronoun_distribution,
            "mean_sentiment": mean_sentiment_all,
            "sentiment_by_pronoun": sentiment_by_pronoun,
            "top_pmi_adj": top_pmi_adj,
            "top_pmi_noun": top_pmi_noun,
            "top_pmi_verb": top_pmi_verb,
            "top_pmi_adj_pronoun_distribution": top_pmi_adj_pronoun_distribution,
            "top_pmi_noun_pronoun_distribution": top_pmi_noun_pronoun_distribution,
            "top_pmi_verb_pronoun_distribution": top_pmi_verb_pronoun_distribution,
            **group_metrics
        }

    except Exception as e:
        print(f"[ERROR] Failed to generate individual report: {e}")
        return {}


def compile_from_individual_reports(reports):
    """
    Aggregate gender-related metrics from multiple individual article-level reports.

    This function combines a list of dictionaries (each created by `generate_individual_report`) into:
    1. A global summary (`aggregated_report`) with corpus-level totals, averages, and top PMI terms.
    2. A set of `individual_values` for each metric, preserving per-document values for statistical visualisation.

    It computes:
    - Corpus-wide totals for actors, mentions, and language flags
    - Aggregated counts and distributions by pronoun group (she/her, he/him)
    - Top PMI adjectives, nouns, and verbs (overall and per pronoun group)
    - Mean, median, and standard deviation for all numeric metrics
    - Sentiment averages overall and by gender group
    - Combined frequency counts for grammatical roles and speech attribution

    Args:
        reports (list of dict): A list of individual article reports, each returned by `generate_individual_report`.

    Returns:
        tuple:
            - aggregated_report (dict): Summary of corpus-wide metrics including:
                * Total counts (e.g. actors, mentions)
                * Sentiment statistics
                * Coded language use
                * Grammatical and referential roles
                * Top PMI words per POS and gender group
            - individual_values (dict): Per-text lists for each metric, to support plotting or statistical analysis.
    """

    reports = [r for r in reports if isinstance(r, dict) and "total_actors" in r]

    accumulated_data = defaultdict(int)
    pmi_adj_words = Counter()
    pmi_noun_words = Counter()
    pmi_verb_words = Counter()
    top_pmi_pronoun = {
        "adj": {"she_her": Counter(), "he_him": Counter()},
        "noun": {"she_her": Counter(), "he_him": Counter()},
        "verb": {"she_her": Counter(), "he_him": Counter()},
    }
    sentiment_by_pronoun = {"she_her": [], "he_him": []}
    individual_values = defaultdict(list)

    for report in reports:
        # General flags and totals
        accumulated_data["total_actors"] += report.get("total_actors", 0)
        accumulated_data["total_mentions"] += report.get("total_mentions", 0)
        accumulated_data["uses_gender_neutral_language"] += int(report.get("uses_gender_neutral_language", False))
        accumulated_data["generic_masculine"] += int(report.get("generic_masculine", False))

        # Simple numeric aggregations
        for pronoun in ["she_her", "he_him"]:
            for field in [
                "pronoun_distribution",
                "mentions_pronoun_distribution",
                "feminine_coded_words_pronoun_distribution",
                "masculine_coded_words_pronoun_distribution",
            ]:
                value = report.get(field, {}).get(pronoun, 0)
                key = f"{field}_{pronoun}"
                accumulated_data[key] += value
                individual_values[key].append(value)

            for field in [
                "name_count", "pronoun_count", "subject_count", "object_count",
                "direct_quote_count", "indirect_quote_count"
            ]:
                value = report.get(f"{field}_{pronoun}", None)
                if value is not None:
                    individual_values[f"{field}_{pronoun}"].append(value)
                    if isinstance(value, numbers.Number):
                        accumulated_data[f"{field}_{pronoun}"] += value

        # PMI words
        pmi_adj_words.update(dict(report.get("top_pmi_adj", [])))
        pmi_noun_words.update(dict(report.get("top_pmi_noun", [])))
        pmi_verb_words.update(dict(report.get("top_pmi_verb", [])))

        for pos in ["adj", "noun", "verb"]:
            for pronoun in ["she_her", "he_him"]:
                top_pmi_pronoun[pos][pronoun].update(
                    dict(report.get(f"top_pmi_{pos}_pronoun_distribution", {}).get(pronoun, []))
                )

        # Sentiment
        sentiment_by_pronoun["she_her"].append(report.get("sentiment_by_pronoun", {}).get("she_her", 0))
        sentiment_by_pronoun["he_him"].append(report.get("sentiment_by_pronoun", {}).get("he_him", 0))
        individual_values["mean_sentiment_all"].append(report.get("mean_sentiment", 0))

        # Global totals
        individual_values["total_actors"].append(report.get("total_actors", 0))
        individual_values["total_mentions"].append(report.get("total_mentions", 0))
        individual_values["total_feminine_coded_words"].append(report.get("total_feminine_coded_words", 0))
        individual_values["total_masculine_coded_words"].append(report.get("total_masculine_coded_words", 0))
        individual_values["uses_gender_neutral_language"].append(int(report.get("uses_gender_neutral_language", False)))
        individual_values["generic_masculine"].append(int(report.get("generic_masculine", False)))

    # Add overall values for count metrics
    for base in [
        "name_count", "pronoun_count", "subject_count", "object_count",
        "direct_quote_count", "indirect_quote_count"
    ]:
        she_val = accumulated_data.get(f"{base}_she_her", 0)
        he_val = accumulated_data.get(f"{base}_he_him", 0)
        accumulated_data[f"{base}_overall"] = she_val + he_val

    # Compute mean/median/std 
    df = pd.DataFrame(individual_values)
    means = df.mean(numeric_only=True).to_dict()
    medians = df.median(numeric_only=True).to_dict()
    stds = df.std(numeric_only=True).to_dict()

    # Final aggregated report
    aggregated_report = {
        "texts_with_actors": len(individual_values["total_actors"]),
        "uses_gender_neutral_language": accumulated_data["uses_gender_neutral_language"],
        "generic_masculine": accumulated_data["generic_masculine"],
        "total_actors": accumulated_data["total_actors"],
        "total_mentions": accumulated_data["total_mentions"],
        "mean_metrics": means,
        "median_metrics": medians,
        "std_metrics": stds,
        "top_pmi_adj": pmi_adj_words.most_common(10),
        "top_pmi_noun": pmi_noun_words.most_common(10),
        "top_pmi_verb": pmi_verb_words.most_common(10),
        "top_pmi_adj_pronoun_distribution": {k: v.most_common(10) for k, v in top_pmi_pronoun["adj"].items()},
        "top_pmi_noun_pronoun_distribution": {k: v.most_common(10) for k, v in top_pmi_pronoun["noun"].items()},
        "top_pmi_verb_pronoun_distribution": {k: v.most_common(10) for k, v in top_pmi_pronoun["verb"].items()},
        "sentiment_by_pronoun": {
            "she_her": (sum(sentiment_by_pronoun["she_her"]) / len(sentiment_by_pronoun["she_her"]))
            if sentiment_by_pronoun["she_her"] else None,
            "he_him": (sum(sentiment_by_pronoun["he_him"]) / len(sentiment_by_pronoun["he_him"]))
            if sentiment_by_pronoun["he_him"] else None,
        },
        "mean_sentiment_all": (
            sum(individual_values["mean_sentiment_all"]) / len(individual_values["mean_sentiment_all"])
            if individual_values["mean_sentiment_all"] else 0.0
        )
    }

    # Add all count metrics to aggregated_report
    for k, v in accumulated_data.items():
        aggregated_report[k] = v

    return aggregated_report, dict(individual_values)


def compile_aggregated_report_batched(temp_dir):
    """
    Compile a corpus-level aggregated report from per-article analysis results.

    This function loads and combines individual actor-level knowledgebases and corresponding
    document-level flags (e.g. gender-neutral language use, generic masculine) to produce:
    - A summary of global gender representation and sentiment metrics
    - Per-text values for statistical analysis and visualisation

    Expected files in the input directory:
        - `knowledgebase_<article_id>.pkl`: Actor-level analysis for one article
        - `flags_<article_id>.pkl`: Document-level metadata flags (e.g. GNL, GM)

    Args:
        temp_dir (Path or str): Directory containing `.pkl` knowledgebase and flag files.

    Returns:
        tuple:
            - aggregated_report (dict): Corpus-wide totals, averages, PMI rankings, and sentiment metrics.
            - individual_values (dict): Per-document metric lists for downstream visualisation and statistics.

    Notes:
        - Articles missing either file are skipped with a warning.
        - Internally uses `generate_individual_report` and `compile_from_individual_reports`.
    """

    temp_dir = Path(temp_dir)
    kb_files = list(temp_dir.glob("knowledgebase_*.pkl"))
    meta_data_files = list(temp_dir.glob("meta_data_*.pkl"))

    if not kb_files or not meta_data_files:
        print("[WARNING] No knowledgebase or flags files found.")
        return {}, {}

    all_reports = []
    print(f"[INFO] Found {len(kb_files)} knowledgebase and {len(meta_data_files)} flag files...")

    for kb_path in sorted(kb_files):
        article_id = kb_path.stem.replace("knowledgebase_", "")
        meta_data_path = temp_dir / f"meta_data_{article_id}.pkl"

        if not meta_data_path.exists():
            print(f"[WARNING] Missing meta data file for {article_id}, skipping.")
            continue

        try:
            kb_df = pd.read_pickle(kb_path)
            with open(meta_data_path, "rb") as f:
                meta_data = pickle.load(f)

            gnl = meta_data.get("uses_gender_neutral_language", False)
            gm = meta_data.get("generic_masculine", False)

            report = generate_individual_report(kb_df, gnl, gm)
            if report:
                all_reports.append(report)

        except Exception as e:
            print(f"[ERROR] Failed to process article {article_id}: {e}")
            continue

    return compile_from_individual_reports(all_reports)


def visualise_boxplot(data, title, xlabel, output_file, exclude_outliers=True, xlim=None):
    """
    Create and save a horizontal boxplot visualising numeric distributions across groups.

    This function is typically used to compare gender-related metrics (e.g. sentiment,
    mentions, coded word counts) between groups like "she/her" and "he/him".

    Features:
    - Optionally removes the top 10 values per group to reduce skew from outliers.
    - Uses seaborn for consistent and clean visual styling.
    - Saves output as a high-resolution PNG file.

    Args:
        data (dict): Dictionary mapping group names (e.g. 'she_her', 'he_him') to lists of numeric values.
        title (str): Plot title displayed at the top of the figure.
        xlabel (str): Label for the x-axis.
        output_file (str): Full path (including filename) to save the output PNG file.
        exclude_outliers (bool): If True (default), removes the top 10 values from each group before plotting.
        xlim (tuple or None): Optional x-axis limits as a (min, max) tuple. If None, determined automatically.

    Output:
        Saves a horizontal boxplot to the specified file path. If all groups are empty after filtering,
        the plot is skipped and a warning is printed.

    Example:
        data = {"she_her": [0.1, 0.2, 0.3], "he_him": [0.5, 0.6, 0.7]}
        visualise_boxplot(data, "Sentiment Distribution", "Sentiment Score", "sentiment.png")
    """

    def remove_outliers(values):
        """
        Helper function to remove the top 10 outliers from a list of data.
        """
        filtered_values = [v for v in values if v is not None]
        if len(filtered_values) > 10:
            return sorted(filtered_values)[:-10]
        return filtered_values

    # Process data based on outlier exclusion setting
    if exclude_outliers:
        filtered_data = {key: remove_outliers(values) for key, values in data.items()}
    else:
        filtered_data = {key: [v for v in values if v is not None] for key, values in data.items()}

    # Remove empty groups
    filtered_data = {key: values for key, values in filtered_data.items() if values}

    if not filtered_data:
        print(f"[WARNING] All groups are empty after filtering; skipping plot: {title}")
        return

    # Prepare data for seaborn
    plt.figure(figsize=(20, 15))
    df = pd.DataFrame({key: pd.Series(values) for key, values in filtered_data.items()}).melt(var_name="Group",
                                                                                              value_name="Values")

    palette = sns.color_palette("Blues", n_colors=len(df["Group"].unique()))
    sns.boxplot(data=df, x="Values", y="Group", hue="Group", orient="h", showfliers=False, palette=palette)

    plt.title(title, fontsize=40)
    plt.xlabel(xlabel, fontsize=35)
    plt.ylabel('Pronouns', fontsize=35)

    # Set X-axis limits if provided
    if xlim:
        plt.xlim(xlim)

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)

    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"[INFO] Saved boxplot to {output_file}.")


def visualise_barplot(data, title, xlabel, output_file):
    """
    Create and save a barplot visualising the frequency of boolean values (True/False).

    This function is typically used to display how many documents in a corpus
    exhibit a binary feature such as gender-neutral language usage or generic masculine usage.

    Features:
    - Counts occurrences of True and False in the input list.
    - Uses seaborn to generate a clean, colour-coded barplot.
    - Saves the plot as a high-resolution PNG image.

    Args:
        data (list of bool): Boolean values representing the presence or absence of a feature across documents.
        title (str): Title displayed at the top of the plot.
        xlabel (str): Label for the x-axis (typically the feature name).
        output_file (str): Full path to save the resulting PNG plot.

    Output:
        A barplot is saved to the specified file path.
        Prints a confirmation message when saved.

    Example:
        visualise_barplot(
            data=[True, False, False, True, True],
            title="Gender-Neutral Language Usage",
            xlabel="Uses GNL",
            output_file="gnl_barplot.png"
        )
    """
    # Count occurrences of True and False
    counts = {"True": data.count(True), "False": data.count(False)}

    # Create the barplot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), hue=list(counts.keys()), palette="Blues")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()
    print(f"[INFO] Saved barplot to {output_file}.")


def plot_group_distribution_from_aggregated(individual_values, metric_pairs, title, output_file):
    """
    Generate a grouped barplot showing the percentage distribution of two metrics by pronoun group.

    This function compares the relative distribution of two related count-based metrics
    (e.g. subject vs object roles, direct vs indirect quotes, named vs pronoun mentions)
    for "she_her" and "he_him" groups. Percentages are calculated per group and plotted side by side.

    Features:
    - Aggregates raw counts across all documents from the `individual_values` dictionary.
    - Computes the proportion of each metric relative to its pair (per group).
    - Produces a horizontal barplot with grouped bars by gender and metric category.
    - Useful for visualising structural or linguistic asymmetries between gender groups.

    Args:
        individual_values (dict): Dictionary of per-text metric lists, expected to contain keys like:
                                  "<metric>_she_her" and "<metric>_he_him" for each item in `metric_pairs`.
        metric_pairs (tuple): A pair of metric name prefixes (e.g. ("subject_count", "object_count")).
        title (str): Title for the generated plot.
        output_file (str): File path to save the resulting PNG image.

    Output:
        Saves a horizontal grouped barplot as a PNG file to `output_file`.
        If no valid data is found, the plot is skipped and a warning is printed.

    Example:
        plot_group_distribution_from_aggregated(
            individual_values,
            metric_pairs=("subject_count", "object_count"),
            title="Subject vs Object Roles",
            output_file="subject_vs_object.png"
        )
    """
    data = []
    for group in ["she_her", "he_him"]:
        a_list = individual_values.get(f"{metric_pairs[0]}_{group}", [])
        b_list = individual_values.get(f"{metric_pairs[1]}_{group}", [])
        a, b = sum(a_list), sum(b_list)
        total = a + b
        if total > 0:
            data.append({"Group": group, "Category": metric_pairs[0].replace("_count", "").replace("_", " ").title(),
                         "Value": a / total * 100})
            data.append({"Group": group, "Category": metric_pairs[1].replace("_count", "").replace("_", " ").title(),
                         "Value": b / total * 100})

    if not data:
        print(f"[WARNING] No data available for plot: {title}")
        return

    df = pd.DataFrame(data)

    plt.figure(figsize=(6, 4))
    palette = sns.color_palette("Blues", n_colors=len(df["Group"].unique()))
    sns.barplot(data=df, x="Value", y="Category", hue="Group", palette=palette)
    plt.title(title)
    plt.xlabel("Percentage")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()


def visualise_aggregated_report(individual_values, output_dir):
    """
    Generate and save visualisations of gender representation metrics from aggregated per-document data.

    This function produces multiple visualisations to summarise key linguistic and structural patterns,
    grouped by gendered pronoun reference ("she/her" vs. "he/him"). It uses the metrics collected during
    corpus-level analysis to show how gender is reflected in mentions, roles, sentiment, and binary stylistic markers.

    Generated visualisations include:
    - Boxplots for:
        * Pronoun distributions
        * Mention counts (name/pronoun)
        * Sentiment scores (overall and by group)
    - Barplots for:
        * Use of gender-neutral language
        * Use of generic masculine
    - Grouped barplots (percentage-based) for:
        * Named vs. pronoun mentions
        * Subject vs. object grammatical roles
        * Direct vs. indirect quotations

    Args:
        individual_values (dict): Dictionary of per-text metric lists (from `compile_from_individual_reports()`),
                                  including gender-coded counts, sentiment scores, grammatical roles, and flags.
        output_dir (str): Directory path where all resulting PNG plots will be saved.

    Output:
        - Saves all plots as high-resolution PNG files in `output_dir`.
        - Prints an info message once all visualisations are saved.

    Notes:
        - Outliers are excluded in boxplots to improve readability.
        - Axes are automatically scaled unless overridden per metric.
        - Plot filenames are generated based on their metric names and type.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Boxplots: Pronouns, Mentions, Sentiment
    pronoun_metrics = {
        "Pronoun Distribution": {
            "total": individual_values["total_actors"],
            "she_her": individual_values.get("pronoun_distribution_she_her", []),
            "he_him": individual_values.get("pronoun_distribution_he_him", [])
        },
        "Mentions by Pronouns": {
            "total": individual_values["total_mentions"],
            "she_her": individual_values.get("mentions_pronoun_distribution_she_her", []),
            "he_him": individual_values.get("mentions_pronoun_distribution_he_him", [])
        },
        "Sentiment by Pronouns": {
            "total": individual_values["mean_sentiment_all"],
            "she_her": individual_values.get("sentiment_by_pronoun_she_her", []),
            "he_him": individual_values.get("sentiment_by_pronoun_he_him", [])
        }
    }

    xlim_ranges = {
        "Pronoun Distribution": (0, 20),
        "Mentions by Pronouns": (0, 50),
        "Sentiment by Pronouns": (-0.27, 0.27)
    }

    for metric, data in pronoun_metrics.items():
        output_file = os.path.join(output_dir, f"{metric.replace(' ', '_').replace('/', '_')}_boxplot.png")
        visualise_boxplot(
            data,
            title=f"{metric} Boxplot",
            xlabel=metric,
            output_file=output_file,
            exclude_outliers=True,
            xlim=xlim_ranges.get(metric),
        )

    # Binary barplots: Gender Neutral and Generic Masculine
    binary_metrics = {
        "Contains Gender Neutral": individual_values["uses_gender_neutral_language"],
        "Generic Masculine": individual_values["generic_masculine"],
    }

    for metric, data in binary_metrics.items():
        output_file = os.path.join(output_dir, f"{metric.replace(' ', '_')}_barplot.png")
        visualise_barplot(
            data,
            title=f"{metric} Barplot",
            xlabel=metric,
            output_file=output_file
        )

    # Grouped distribution plots (percent-based)
    plot_group_distribution_from_aggregated(
        individual_values,
        metric_pairs=("name_count", "pronoun_count"),
        title="Named vs Pronoun Mentions",
        output_file=os.path.join(output_dir, "Name_vs_Pronoun_barplot.png")
    )

    plot_group_distribution_from_aggregated(
        individual_values,
        metric_pairs=("subject_count", "object_count"),
        title="Subject vs Object Roles",
        output_file=os.path.join(output_dir, "Subject_vs_Object_barplot.png")
    )

    plot_group_distribution_from_aggregated(
        individual_values,
        metric_pairs=("direct_quote_count", "indirect_quote_count"),
        title="Direct vs Indirect Quotes",
        output_file=os.path.join(output_dir, "Direct_vs_Indirect_Quotes_barplot.png")
    )

    print(f"[INFO] Boxplots and barplots saved in {output_dir}")


def should_exclude_text(report, sentiment_threshold=0.3, role_threshold=0.5, quote_threshold=0.5, naming_threshold=0.5,
                        min_flags=2):
    """
    Determine whether a text should be excluded due to strong gendered imbalances.

    This function evaluates whether a single article exhibits pronounced representational asymmetries
    between actors associated with "she/her" and "he/him" pronouns. If enough indicators exceed predefined
    thresholds, the article is flagged as discriminatory and marked for exclusion from the corpus.

    Criteria (each computed as absolute difference between gender groups, using Laplace smoothing):
    1. Sentiment Gap:
        Difference in average sentiment score exceeds `sentiment_threshold` (default: 0.3).
    2. Subject/Object Role Gap:
        Difference in subject-to-object ratio exceeds `role_threshold` (default: 0.5).
    3. Quote Type Gap:
        Difference in direct-to-indirect quote ratio exceeds `quote_threshold` (default: 0.5).
    4. Naming vs Pronoun Gap:
        Difference in named-to-pronoun mention ratio exceeds `naming_threshold` (default: 0.5).

    A text is excluded if at least `min_flags` of these 4 imbalance criteria are triggered (default: 2).

    Args:
        report (dict): Gender-aware statistics for a single text (from `generate_individual_report`).
        sentiment_threshold (float): Minimum sentiment gap to trigger the sentiment imbalance flag.
        role_threshold (float): Minimum difference in subject/object ratios to flag imbalance.
        quote_threshold (float): Minimum quote type ratio difference to flag imbalance.
        naming_threshold (float): Minimum difference in naming vs pronoun ratio to flag imbalance.
        min_flags (int): Minimum number of imbalance flags required to mark a text for exclusion.

    Returns:
        tuple:
            - exclude (bool): True if the article meets or exceeds the flag threshold.
            - flags (dict): Dictionary showing which imbalance types were triggered (True/False).
    """
    try:
        she_sent = report["sentiment_by_pronoun"].get("she_her", 0)
        he_sent = report["sentiment_by_pronoun"].get("he_him", 0)

        she_subj = report.get("subject_count_she_her", 0)
        she_obj = report.get("object_count_she_her", 0)
        he_subj = report.get("subject_count_he_him", 0)
        he_obj = report.get("object_count_he_him", 0)

        she_quote_direct = report.get("direct_quote_count_she_her", 0)
        she_quote_indirect = report.get("indirect_quote_count_she_her", 0)
        he_quote_direct = report.get("direct_quote_count_he_him", 0)
        he_quote_indirect = report.get("indirect_quote_count_he_him", 0)

        she_name = report.get("name_count_she_her", 0)
        she_pronoun = report.get("pronoun_count_she_her", 0)
        he_name = report.get("name_count_he_him", 0)
        he_pronoun = report.get("pronoun_count_he_him", 0)

        # Compute robust ratios with +1 smoothing
        subj_ratio_she = (she_subj + 1) / (she_obj + 1)
        subj_ratio_he = (he_subj + 1) / (he_obj + 1)
        subj_ratio_diff = abs(subj_ratio_she - subj_ratio_he)

        quote_ratio_she = (she_quote_direct + 1) / (she_quote_indirect + 1)
        quote_ratio_he = (he_quote_direct + 1) / (he_quote_indirect + 1)
        quote_ratio_diff = abs(quote_ratio_she - quote_ratio_he)

        naming_ratio_she = (she_name + 1) / (she_pronoun + 1)
        naming_ratio_he = (he_name + 1) / (he_pronoun + 1)
        naming_ratio_diff = abs(naming_ratio_she - naming_ratio_he)

        sentiment_diff = abs(she_sent - he_sent)

        flags = {
            "sentiment_gap": sentiment_diff > sentiment_threshold,
            "subject_gap": subj_ratio_diff > role_threshold,
            "quote_gap": quote_ratio_diff > quote_threshold,
            "naming_gap": naming_ratio_diff > naming_threshold,
        }

        return sum(flags.values()) >= min_flags, flags

    except Exception as e:
        print(f"[WARNING] Could not compute exclusion for report: {e}")
        return False, {}


def extract_gender_ratio_summary(knowledgebase_path, summary_dir):
    """
        Extract and save a summary of gendered mention counts from a single knowledgebase file.

        This function reads a pickled actor-level knowledgebase, computes the number of mentions
        attributed to "she/her" and "he/him" pronoun groups, and records the total number of mentions
        and unique actors. The result is saved as a JSON file for use in aggregated visualisation
        and analysis of gender ratios.

        Args:
            knowledgebase_path (Path or str): Path to a knowledgebase .pkl file (one article).
            summary_dir (Path or str): Directory where the resulting summary JSON should be saved.

        Output:
            A JSON file named `summary_<article_id>.json` is saved in `summary_dir`, containing:
                - she_mentions: Total number of "she/her" mentions
                - he_mentions: Total number of "he/him" mentions
                - total_mentions: Combined count of gendered mentions
                - actor_count: Number of unique actors in the article

        Notes:
            - Uses a temporary `.json.tmp` file and atomic rename to avoid partial writes.
            - Skips empty knowledgebases silently.
            - Prints a warning if an error occurs during processing.
        """
    try:
        knowledgebase_dataframe = pd.read_pickle(knowledgebase_path)
        if knowledgebase_dataframe.empty:
            return

        grouped_mentions = knowledgebase_dataframe.groupby("main_pronoun")["mention_count"].sum()
        she_mentions = grouped_mentions.get("she_her", 0)
        he_mentions = grouped_mentions.get("he_him", 0)
        total_mentions = she_mentions + he_mentions
        actor_count = knowledgebase_dataframe["actor_id"].nunique()

        summary = {
            "she_mentions": int(she_mentions),
            "he_mentions": int(he_mentions),
            "total_mentions": int(total_mentions),
            "actor_count": int(actor_count)
        }

        article_id = knowledgebase_path.stem.replace("knowledgebase_", "")
        summary_path = Path(summary_dir) / f"summary_{article_id}.json"
        tmp_path = summary_path.with_suffix(".json.tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(summary, f)

        tmp_path.rename(summary_path)

        print(f"[INFO] Summary saved: {summary_path.name}")

    except Exception as error:
        print(f"[WARNING] Could not process {knowledgebase_path.name}: {error}")

def generate_ratio_summary_parallel(temp_dir, summary_dir):
    """
        Generate gender ratio summaries in parallel for all knowledgebases in a directory.

        This function scans a temporary directory for pickled article-level knowledgebases,
        then extracts gendered mention statistics (she/her vs. he/him) using multiprocessing.
        Each article's summary is saved as a separate JSON file in the specified summary directory.

        Args:
            temp_dir (str or Path): Directory containing `knowledgebase_*.pkl` files (one per article).
            summary_dir (str or Path): Directory where JSON summary files will be stored.

        Output:
            - One `summary_<article_id>.json` per processed article, saved in `summary_dir`.
            - Summary files include:
                * she_mentions
                * he_mentions
                * total_mentions
                * actor_count
            - Prints progress and a final info message on completion.

        Notes:
            - Uses `ProcessPoolExecutor` for parallel processing.
            - Relies on `extract_gender_ratio_summary()` to generate each summary.
            - Skips empty or invalid files silently.
    """
    knowledgebase_files = list(Path(temp_dir).glob("knowledgebase_*.pkl"))
    if not knowledgebase_files:
        print("[WARNING] No knowledgebases found.")
        return

    Path(summary_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting ratio summaries from {len(knowledgebase_files)} knowledgebases...")

    worker_function = partial(extract_gender_ratio_summary, summary_dir=summary_dir)

    with ProcessPoolExecutor() as executor:
        for _ in executor.map(worker_function, knowledgebase_files):
            pass
    print(f"[INFO] Ratio summaries saved to {summary_dir}")

def visualise_gender_ratio_distribution(summary_dir, output_file):
    """
    Create and save a histogram visualising the distribution of gender ratios per article.

    This function processes article-level gender summaries from JSON files and computes
    the percentage of she/her mentions relative to all gendered mentions. It creates two
    side-by-side histograms:
      - One weighted by total number of gendered mentions per article.
      - One weighted by the number of actors per article.

    The gender ratio per article is defined as:
        she_ratio = 100 * she_mentions / (she_mentions + he_mentions)

    This visualisation supports diagnosis of gender skew across the corpus, especially
    before and after applying exclusion or balancing strategies.

    Args:
        summary_dir (Path or str): Directory containing per-article `summary_*.json` files
                                   with mention and actor counts.
        output_file (Path or str): Full file path where the PNG plot will be saved.

    Output:
        - Saves a side-by-side histogram as a PNG file.
        - Left plot: mention-weighted distribution of she/her ratios.
        - Right plot: actor-weighted distribution of she/her ratios.
        - Prints status messages for success or missing data.

    Notes:
        - Skips articles with zero gendered mentions.
        - Uses Seaborn's "Blues" palette for visual clarity.
        - Bin count is fixed at 20 for consistent layout.
    """
    summary_files = list(Path(summary_dir).glob("summary_*.json"))
    if not summary_files:
        print("[WARNING] No ratio summaries found.")
        return

    she_ratios = []
    mention_weights = []
    actor_weights = []

    mpl.rcParams.update({
        "axes.titlesize": 25,
        "axes.labelsize": 25,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "legend.fontsize": 25
    })

    for summary_file in summary_files:
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)

            total_mentions = summary.get("total_mentions", 0)
            she_mentions = summary.get("she_mentions", 0)
            actor_count = summary.get("actor_count", 0)

            if total_mentions > 0:
                she_ratio = (she_mentions / total_mentions) * 100
                she_ratios.append(she_ratio)
                mention_weights.append(total_mentions)
                actor_weights.append(actor_count)

        except Exception as error:
            print(f"[WARNING] Failed to load summary {summary_file.name}: {error}")

    if not she_ratios:
        print("[WARNING] No valid gender ratio summaries found.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bins = 20
    palette = sns.color_palette("Blues", n_colors=7)

    sns.histplot(
        x=she_ratios, bins=bins, weights=mention_weights,
        color=palette[4], stat="percent", ax=axs[0]
    )
    axs[0].set_title("Mention-weighted (Percent)")
    axs[0].set_xlabel("Percentage of she/her Mentions")
    axs[0].set_ylabel("Percentage of Total")
    axs[0].set_ylim(0, 60)

    sns.histplot(
        x=she_ratios, bins=bins, weights=actor_weights,
        color=palette[5], stat="percent", ax=axs[1]
    )
    axs[1].set_title("Actor-weighted (Percent)")
    axs[1].set_xlabel("Percentage of she/her Actors")
    # axs[1].set_ylabel("Percentage of Total Actors")
    axs[1].set_ylim(0, 60)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[INFO] Gender ratio plot saved to {output_file}")

def corpus_level_balance_filter(tmp_dir, exclusion_file_path, equilibrium_lower=0.75, equilibrium_upper=1.25):
    """
    Perform impact-aware corpus-level gender balancing by selectively excluding articles.

    This function identifies and removes articles that overly reinforce a gender imbalance in the corpus
    based on actor and mention ratios. It is typically used after text-level filtering to achieve corpus-wide
    representational equilibrium.

    The procedure:
    1. Loads actor-level and metadata files from a temporary directory.
    2. Calculates the current gender ratios:
        - Mention ratio: she_mentions / he_mentions
        - Actor ratio: she_actors / he_actors
    3. If both ratios fall within the defined equilibrium interval, no further action is taken.
    4. Otherwise, it identifies articles that exclusively contain actors of the overrepresented gender.
       These are ranked by an impact score and excluded until both ratios fall into the target range.
    5. The resulting exclusions are stored in a JSON index structured by year and month.

    Args:
        tmp_dir (Path or str): Directory containing `knowledgebase_*.pkl` and `meta_data_*.pkl` files.
        exclusion_file_path (Path or str): Path to the output `excluded_index.json`.
        equilibrium_lower (float): Minimum acceptable gender ratio (default: 0.75).
        equilibrium_upper (float): Maximum acceptable gender ratio (default: 1.25).

    Output:
        - Updates `excluded_index.json` with excluded article IDs grouped by month.
        - Deletes excluded knowledgebase and metadata files from the `tmp_dir`.
        - Prints detailed progress, ratios, and exclusions to console.

    Notes:
        - Uses an impact score of `mentions + 10 * actors` to prioritise high-impact articles.
        - Applies Laplace smoothing internally when calculating ratios (to prevent divide-by-zero).
        - Only articles containing **only** overrepresented-gender actors are eligible for exclusion.
        - Typical usage: as Step 3 of the balanced corpus generation pipeline.
    """
    tmp_dir = Path(tmp_dir)
    exclusion_file_path = Path(exclusion_file_path)

    print("[INFO] Running impact-aware corpus-level balancing...")

    # Collect article-level statistics
    article_statistics = []
    metadata_by_article = {}

    for knowledgebase_path in tmp_dir.glob("knowledgebase_*.pkl"):
        article_id = knowledgebase_path.stem.replace("knowledgebase_", "")
        metadata_path = tmp_dir / f"meta_data_{article_id}.pkl"
        if not metadata_path.exists():
            continue

        try:
            dataframe = pd.read_pickle(knowledgebase_path)
            if dataframe.empty:
                continue

            she_mentions = dataframe[dataframe.main_pronoun == "she_her"].mention_count.sum()
            he_mentions = dataframe[dataframe.main_pronoun == "he_him"].mention_count.sum()
            she_actors = dataframe[dataframe.main_pronoun == "she_her"].actor_id.nunique()
            he_actors = dataframe[dataframe.main_pronoun == "he_him"].actor_id.nunique()

            article_statistics.append({
                "article_id": article_id,
                "she_mentions": she_mentions,
                "he_mentions": he_mentions,
                "she_actors": she_actors,
                "he_actors": he_actors,
            })

            with open(metadata_path, "rb") as file:
                metadata_by_article[article_id] = pickle.load(file)

        except Exception as error:
            print(f"[WARNING] Skipping {knowledgebase_path.name}: {error}")

    if not article_statistics:
        print("[WARNING] No valid articles found.")
        return

    # Compute totals
    total_she_mentions = sum(article["she_mentions"] for article in article_statistics)
    total_he_mentions = sum(article["he_mentions"] for article in article_statistics)
    total_she_actors = sum(article["she_actors"] for article in article_statistics)
    total_he_actors = sum(article["he_actors"] for article in article_statistics)

    def get_ratio(numerator, denominator):
        return numerator / denominator if denominator > 0 else float('inf')

    initial_mention_ratio = get_ratio(total_she_mentions, total_he_mentions)
    initial_actor_ratio = get_ratio(total_she_actors, total_he_actors)

    print(f"[INFO] Initial mention ratio: {initial_mention_ratio:.3f}, actor ratio: {initial_actor_ratio:.3f}")

    # Early exit
    if equilibrium_lower <= initial_mention_ratio <= equilibrium_upper and equilibrium_lower <= initial_actor_ratio <= equilibrium_upper:
        print("[INFO] Corpus is already balanced. No exclusions applied.")
        return

    # Determine imbalance direction
    imbalance_direction = "male" if initial_mention_ratio < equilibrium_lower or initial_actor_ratio < equilibrium_lower else "female"
    print(f"[INFO] Imbalance direction: too much {imbalance_direction}-representation")

    # Collect exclusion candidates
    exclusion_candidates = []
    for article in article_statistics:
        if imbalance_direction == "male" and article["she_mentions"] == 0 and article["she_actors"] == 0:
            impact_score = article["he_mentions"] + 10 * article["he_actors"]
            exclusion_candidates.append((impact_score, article))
        elif imbalance_direction == "female" and article["he_mentions"] == 0 and article["he_actors"] == 0:
            impact_score = article["she_mentions"] + 10 * article["she_actors"]
            exclusion_candidates.append((impact_score, article))

    exclusion_candidates.sort(reverse=True, key=lambda x: x[0])

    # Simulate exclusion
    excluded_article_ids = set()
    for _, article in exclusion_candidates:
        total_she_mentions -= article["she_mentions"]
        total_he_mentions -= article["he_mentions"]
        total_she_actors -= article["she_actors"]
        total_he_actors -= article["he_actors"]

        excluded_article_ids.add(article["article_id"])

        current_mention_ratio = get_ratio(total_she_mentions, total_he_mentions)
        current_actor_ratio = get_ratio(total_she_actors, total_he_actors)

        if equilibrium_lower <= current_mention_ratio <= equilibrium_upper and equilibrium_lower <= current_actor_ratio <= equilibrium_upper:
            print("[INFO] Equilibrium reached.")
            break

    # Write exclusion index
    exclusion_index = {}
    for article_id in excluded_article_ids:
        metadata = metadata_by_article.get(article_id, {})
        year = str(metadata.get("year"))
        month = str(metadata.get("month")).zfill(2)
        month_key = f"{year}-{month}"
        exclusion_index.setdefault(month_key, []).append(article_id)

    with open(exclusion_file_path, "w", encoding="utf-8") as file:
        json.dump(exclusion_index, file, indent=2)

    # Delete excluded knowledgebase and metadata files
    for article_id in excluded_article_ids:
        for prefix in ["knowledgebase_", "meta_data_"]:
            file_path = tmp_dir / f"{prefix}{article_id}.pkl"
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"[INFO] Deleted: {file_path.name}")
                except Exception as e:
                    print(f"[WARNING] Could not delete {file_path.name}: {e}")

    print(f"[INFO] Exclusion complete. Removed {len(excluded_article_ids)} articles.")
    print(f"[INFO] Final mention ratio: {current_mention_ratio:.3f}, actor ratio: {current_actor_ratio:.3f}")


def filter_existing_knowledgebases(
        input_dir,
        output_dir,
        exclusion_file,
        sentiment_threshold=0.3,
        role_threshold=0.5,
        quote_threshold=0.5,
        naming_threshold=0.5,
        min_flags=2
):
    """
    Filter precomputed knowledgebases for discriminatory patterns based on actor-level metrics.

    This function evaluates each article’s knowledgebase using multiple heuristics for gender-based
    representational imbalances. Articles that trigger a configurable number of these heuristics
    are excluded from the dataset to reduce distortion in downstream analyses.

    The heuristics are based on asymmetries between she/her and he/him actors:
      1. Sentiment Gap: Mean sentiment differs by more than `sentiment_threshold`.
      2. Subject/Object Role Gap: Subject-to-object ratio differs by more than `role_threshold`.
      3. Quote Type Gap: Ratio of direct to indirect quotes differs by more than `quote_threshold`.
      4. Naming Gap: Named vs pronoun mentions differ by more than `naming_threshold`.

    Articles are excluded only if at least `min_flags` of the above imbalances are triggered.

    For each excluded article:
      - The actor-level file is deleted.
      - The corresponding flag breakdown is saved as `flags_<article_id>.pkl` in the output directory.
      - The article ID is logged into a monthly exclusion index JSON file.

    Args:
        input_dir (Path or str): Directory containing `knowledgebase_*.pkl` and `meta_data_*.pkl` files.
        output_dir (Path or str): Destination for retained metadata and per-article flags.
        exclusion_file (Path or str): Path to `excluded_index.json` to store structured exclusion metadata.
        sentiment_threshold (float): Maximum allowed sentiment difference between groups (default: 0.3).
        role_threshold (float): Maximum allowed difference in subject-to-object role ratios (default: 0.5).
        quote_threshold (float): Maximum allowed difference in quote type ratios (default: 0.5).
        naming_threshold (float): Maximum allowed difference in name-to-pronoun ratios (default: 0.5).
        min_flags (int): Number of violated criteria needed to trigger exclusion (default: 2).

    Returns:
        dict: A month-wise dictionary of excluded article IDs (structured as {"YYYY-MM": [ids]}).

    Notes:
        - Articles without a corresponding `meta_data_*.pkl` file are skipped.
        - Remaining (non-excluded) articles remain untouched in the input directory.
        - Intended for use in Step 2 (Text-level Filtering) of the discrimination-aware corpus balancing pipeline.
    """
    input_directory = Path(input_dir)
    output_directory = Path(output_dir)
    exclusion_file_path = Path(exclusion_file)
    output_directory.mkdir(parents=True, exist_ok=True)

    excluded_index = {}

    for knowledgebase_path in sorted(input_directory.glob("knowledgebase_*.pkl")):
        article_id = knowledgebase_path.stem.replace("knowledgebase_", "")
        meta_path = input_directory / f"meta_data_{article_id}.pkl"

        if not meta_path.exists():
            print(f"[WARNING] Missing flags file for article ID {article_id}, skipping.")
            continue

        try:
            knowledgebase_dataframe = pd.read_pickle(knowledgebase_path)
            with open(meta_path, "rb") as file:
                meta_data = pickle.load(file)

            uses_gender_neutral_language = meta_data.get("uses_gender_neutral_language", False)
            uses_generic_masculine = meta_data.get("generic_masculine", False)

            individual_report = generate_individual_report(
                knowledgebase_dataframe,
                uses_gender_neutral_language=uses_gender_neutral_language,
                generic_masculine=uses_generic_masculine
            )

            should_exclude, flags = should_exclude_text(
                individual_report,
                sentiment_threshold=sentiment_threshold,
                role_threshold=role_threshold,
                quote_threshold=quote_threshold,
                naming_threshold=naming_threshold,
                min_flags=min_flags
            )

            year = str(meta_data.get("year"))
            month = str(meta_data.get("month")).zfill(2)
            month_key = f"{year}-{month}"

            if should_exclude:
                excluded_index.setdefault(month_key, []).append(article_id)

                with open(output_directory / f"flags_{article_id}.pkl", "wb") as flag_file:
                    pickle.dump(flags, flag_file)

                try:
                    knowledgebase_path.unlink()
                    print(f"[INFO] Deleted excluded KB: {knowledgebase_path.name}")
                except Exception as e:
                    print(f"[WARNING] Could not delete {knowledgebase_path.name}: {e}")

        except Exception as error:
            print(f"[WARNING] Failed to process article {article_id}: {error}")
            continue

    with open(exclusion_file_path, "w", encoding="utf-8") as file:
        json.dump(excluded_index, file, indent=2)

    print(f"[INFO] Filtering complete. Excluded {sum(len(value) for value in excluded_index.values())} article(s).")

    return excluded_index


def clear_directory(directory_path):
    """
        Delete all files from the specified directory. Create the directory if it does not exist.

        This function is typically used to reset a working directory before writing new intermediate
        or output files. It does not delete subdirectories or recurse into nested folders—only top-level files
        are removed.

        Args:
            directory_path (Path or str): Path to the directory to be cleared.

        Behaviour:
            - If the directory exists: all files in the directory are deleted.
            - If the directory does not exist: it is created (including any missing parent directories).
    """
    directory = Path(directory_path)
    if directory.exists():
        for file in directory.glob("*"):
            file.unlink()
    else:
        directory.mkdir(parents=True, exist_ok=True)


def process_chunk(job_chunk):
    """
    Process and filter monthly corpus files by excluding flagged articles.

    Each chunk contains one or more monthly files, each with a set of article IDs to exclude.
    For each file in the chunk, the function:
      - Loads the JSON file containing article texts (format: taz_YYYY-MM.json)
      - Removes all articles whose IDs appear in the provided exclusion list
      - Writes the filtered articles to the specified output path in a year-based directory structure

    This function is designed for parallel execution across multiple months.

    Args:
        job_chunk (list of tuples): Each tuple has the form:
            (
                input_file (Path): Path to the original monthly JSON file (taz_YYYY-MM.json),
                excluded_ids (set): Set of article IDs (as strings) to exclude from the file,
                output_file (Path): Path to write the filtered JSON output (preserves format)
            )

    Returns:
        tuple:
            - total_orig (int): Total number of articles across all input files before filtering
            - total_filter (int): Total number of articles remaining after filtering
    """
    total_orig, total_filter = 0, 0

    for input_file, excluded_ids, output_file in job_chunk:
        try:
            with open(input_file, "r", encoding="utf-8") as file:
                articles = json.load(file)
        except Exception as e:
            print(f"[ERROR] Failed to read {input_file}: {e}")
            continue

        before = len(articles)
        filtered = {article_id: article for article_id, article in articles.items() if str(article_id) not in excluded_ids}
        after = len(filtered)

        try:
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(filtered, file, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write {output_file}: {e}")
            continue

        print(f"[INFO] {input_file.name}: {before} → {after}")
        total_orig += before
        total_filter += after

    return total_orig, total_filter


def create_balanced_corpus_parallel(corpus_dir, exclusion_index_path, output_dir, max_workers=5):
    """
    Create a filtered, balanced version of the original corpus based on an exclusion index.

    This function reads an exclusion index (mapping months to article IDs),
    loads the corresponding monthly article files, removes excluded articles,
    and writes the filtered content to a new corpus directory — preserving the original structure.

    The filtering is done in parallel using multiple worker processes.

    Args:
        corpus_dir (Path or str): Path to the input corpus root directory (e.g. 'corpus/').
                                  Each subdirectory should be named by year (e.g. '1980'),
                                  containing files like 'taz_1980-01.json'.
        exclusion_index_path (Path or str): Path to the exclusion file (JSON format) that maps
                                            'YYYY-MM' keys to lists of article IDs to exclude.
        output_dir (Path or str): Target directory to write the balanced corpus (e.g. 'balanced_corpus/').
                                  Directory structure (by year) will be preserved.
        max_workers (int, optional): Number of parallel processes to use. Default is 5.

    Exclusion Index Format:
        A JSON file mapping months to article IDs:
        {
            "1980-01": ["article123", "article456"],
            "1980-02": ["article789"],
            ...
        }

    Outputs:
        - Filtered JSON files saved to `output_dir/year/taz_YYYY-MM.json`
        - Summary printed to console with total articles before and after filtering

    Notes:
        - Skips any month not found in the input corpus.
        - Uses multiprocessing for efficient large-scale filtering.
        - Article IDs are expected to be strings.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(exclusion_index_path, "r", encoding="utf-8") as file:
            exclusion_index = json.load(file)
    except Exception as e:
        print(f"[ERROR] Could not load exclusion index from {exclusion_index_path}: {e}")
        return

    print(f"[INFO] Loaded exclusion index with {len(exclusion_index)} affected months.")

    all_jobs = []
    for year_month, excluded_ids in exclusion_index.items():
        year, month = year_month.split("-")
        input_file = corpus_dir / year / f"taz_{year}-{month}.json"
        output_file = output_dir / year / f"taz_{year}-{month}.json"

        if not input_file.exists():
            print(f"[WARNING] File not found: {input_file}")
            continue

        output_file.parent.mkdir(parents=True, exist_ok=True)
        all_jobs.append((input_file, set(excluded_ids), output_file))

    if not all_jobs:
        print("[INFO] No files to process.")
        return

    chunk_size = math.ceil(len(all_jobs) / max_workers)
    job_chunks = [all_jobs[i:i + chunk_size] for i in range(0, len(all_jobs), chunk_size)]

    total_original, total_filtered = 0, 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in job_chunks]
        for future in futures:
            try:
                orig, filt = future.result()
                total_original += orig
                total_filtered += filt
            except Exception as e:
                print(f"[ERROR] Worker failed: {e}")

    print(f"[INFO] Balanced corpus saved to: {output_dir.resolve()}")
    print(f"[INFO] Total articles before filtering: {total_original}")
    print(f"[INFO] Total articles after filtering: {total_filtered}")


def save_exclusion_report(step1_count, step2_excluded, step2_flags, step3_excluded, output_file):
    """
        Write a plain-text summary report of article exclusions during multi-step filtering.

        This report documents how many articles were processed and excluded during each step
        of the corpus filtering pipeline:
          - Step 1: Initial processing (total number of articles considered)
          - Step 2: Discriminatory content exclusion based on heuristic rules
          - Step 3: Corpus-level gender balance filtering

        It also includes a breakdown of which discrimination flags were triggered
        and how frequently they contributed to exclusions in Step 2.

        Args:
            step1_count (int): Total number of articles initially processed.
            step2_excluded (int): Number of articles excluded during Step 2 (heuristic discrimination filter).
            step2_flags (dict): Dictionary with flag names as keys and counts as values.
                                For example: {"sentiment_gap": 25, "quote_gap": 13}
            step3_excluded (int): Number of articles excluded during Step 3 (corpus-level balancing).
            output_file (str or Path): Path to the `.txt` file where the report will be saved.

        Output:
            Saves a formatted `.txt` report to the specified file.
            Prints an info message on success or an error message if the write fails.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Exclusion Report\n")
            f.write("================\n\n")
            f.write(f"Step 1 - Total articles processed: {step1_count}\n\n")
            f.write("Step 2 - Discriminatory Text Filtering:\n")
            f.write(f"  Articles excluded: {step2_excluded}\n")
            f.write("  Flags triggered (sum across excluded):\n")
            for flag, count in step2_flags.items():
                f.write(f"    {flag}: {count}\n")
            f.write("\n")
            f.write("Step 3 - Corpus-level Balancing:\n")
            f.write(f"  Articles excluded: {step3_excluded}\n")
        print(f"[INFO] Exclusion report saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to write exclusion report: {e}")
