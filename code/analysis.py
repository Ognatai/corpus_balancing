import pandas as pd
import math
from collections import Counter
import re
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")

generic_woman = {
    "frau", "frauen", "mutter", "mütter", "mama", "oma", "großmutter", "dame", "damen"
}

generic_man = {
    "mann", "männer", "vater", "väter", "papa", "opa", "großvater", "herr", "herren"
}

generic_terms = generic_woman | generic_man

masculine_coded_words = {
    "abenteuer", "aggressiv", "ambition", "analytisch", "aufgabenorientiert", "autark", "autoritär", "autonom",
    "beharr", "besieg", "bestimmt", "direkt", "domin", "durchsetz", "ehrgeiz", "eigenständig", "einfluss", "einflussreich",
    "energisch", "entscheid", "entschlossen", "erfolgsorientiert", "führ", "gewinn", "hartnäckig", "herausfordern",
    "hierarch", "kompetitiv", "konkurrenz", "kräftig", "kraft", "leisten", "leistungsfähig", "leistungsorientiert", "leit",
    "lenken", "mutig", "offensiv", "persisten", "rational", "risiko", "selbstbewusst", "selbstsicher", "selbstständig",
    "selbstvertrauen", "stark", "stärke", "stolz", "überlegen", "unabhängig", "wettbewerb", "wetteifer", "wettkampf",
    "wettstreit", "willens", "zielorientiert", "zielsicher", "zielstrebig"
}

feminine_coded_words = {
    "angenehm", "aufrichtig", "beraten", "bescheiden", "betreu", "beziehung", "commit", "dankbar", "ehrlich", "einfühl",
    "emotion", "empath", "engag", "familie", "fleiß", "förder", "freundlich", "freundschaft", "fürsorg", "gefühl",
    "gemeinsam", "gemeinschaft", "gruppe", "harmon", "helfen", "herzlich", "hilf", "höflich", "interpers", "kollabor",
    "kollegial", "kooper", "kümmern", "liebenswürdig", "loyal", "miteinander", "mitfühl", "mitgefühl", "nett",
    "partnerschaftlich", "pflege", "rücksicht", "sensibel", "sozial", "team", "treu", "umgänglich", "umsichtig",
    "uneigennützig", "unterstütz", "verantwortung", "verbunden", "verein", "verlässlich", "verständnis", "vertrauen",
    "wertschätz", "zugehörig", "zusammen", "zuverlässig", "zwischenmensch"
}

indirect_quote_verbs = {
    "sagen", "meinen", "behaupten", "erklären", "berichten", "betonen", "erläutern", "darlegen", "feststellen", "ergänzen", 
    "bemerken", "erwidern", "anmerken", "zugeben", "einräumen", "andeuten", "schildern", "informieren", "äußern", 
    "hinweisen", "anführen", "ankündigen", "kommentieren", "beobachten", "kritisieren", "fordern", "warnen", "rügen", 
    "bestätigen", "verneinen", "verlautbaren", "einwenden"
}



# === main processing  ===

def process_single_text(doc, stopwords):
    """
    Extract actor-level annotations and gender-related metrics from a SpaCy-parsed document.

    This function performs the core analysis on a single article:
    - Detects named and generic actors (e.g. "Angela Merkel", "die Frau").
    - Resolves coreferent pronouns using Coreferee.
    - Assigns each actor a unique ID and a dominant pronoun group ('she_her'or 'he_him').
    - Extracts predications (sentences mentioning the actor) and annotates them with:
        * Grammatical role (subject/object/context)
        * Quote type (direct/indirect speech)
        * Sentiment score (based on BERT model)
    - Computes:
        * Gender-coded language counts (feminine and masculine-coded stems)
        * PMI (Pointwise Mutual Information) keywords for adjectives, nouns, and verbs
    - Detects whether the overall text uses:
        * Gender-neutral language (e.g. "*innen", "Studierende")
        * Generic masculine forms (capitalised gendered nouns without inclusive markers)

    Args:
        doc (spacy.tokens.Doc): A parsed SpaCy document including coreference annotations.
        stopwords (set): Set of stopwords used for filtering PMI candidates.

    Returns:
        tuple:
            - pd.DataFrame: One row per actor with linguistic and gender-based annotations.
            - bool: Whether the text uses gender-neutral language.
            - bool: Whether the text exhibits generic masculine usage.
    """
    
    base_actors = get_actors(doc)
    base_actors = combine_names(base_actors)
    base_actors = get_generic_names(doc, base_actors)
    pronouns = detect_pronouns(doc, base_actors)
    actor_ids = generate_actor_ids(base_actors)
    
    contains_gnl = detect_gender_neutral_language(doc)
    generic_masc = detect_generic_masculine(doc)

    actor_rows = []
    
    for actor_name, nomination_tokens in base_actors.items():
        actor_id = actor_ids[actor_name]
        pronoun_tokens = pronouns.get(actor_name, [])
        main_pronoun_group = detect_pronoun_group([t.text for t in pronoun_tokens])
        
        if not main_pronoun_group:
            continue  # skip actors without a clear pronoun group

        predications = extract_predications(actor_id, actor_name, nomination_tokens, pronoun_tokens)
        predication_texts = [p["predication_text"] for p in predications]
        
        all_text = " ".join(predication_texts)
        
        uttered_by_actor = [p["predication_text"] for p in predications if p["uttered_by_person"]]
        direct_quotes = sum(p["uttered_by_person"] for p in predications)
        indirect_quotes = sum(p["is_indirect_quote"] for p in predications)
        subject_role = sum(p["syntactic_role"] == "nsubj" for p in predications)
        object_role = sum(p["syntactic_role"] == "obj" for p in predications)
        
        sentiments = [detect_sentiment_score(p["predication_text"]) for p in predications]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        feminine_words = count_gendered_stems(all_text, feminine_coded_words)
        masculine_words = count_gendered_stems(all_text, masculine_coded_words)

        pmi_adj = compute_pmi(nomination_tokens + pronoun_tokens, doc, stopwords, pos="ADJ")
        pmi_noun = compute_pmi(nomination_tokens + pronoun_tokens, doc, stopwords, pos="NOUN")
        pmi_verb = compute_pmi(nomination_tokens + pronoun_tokens, doc, stopwords, pos="VERB")

        actor_rows.append({
            "actor_id": actor_id,
            "actor_nomination": [t.text for t in nomination_tokens],
            "actor_pronouns": [t.text for t in pronoun_tokens],
            "main_pronoun": main_pronoun_group,
            "predication": predication_texts,
            "uttered_by_actor": uttered_by_actor,
            "indirect_quotes": indirect_quotes,
            "direct_quotes": direct_quotes,
            "subject_role": subject_role,
            "object_role": object_role,
            "sentiment": avg_sentiment,
            "feminine_coded_words": feminine_words,
            "masculine_coded_words": masculine_words,
            "pmi_adjective": pmi_adj,
            "pmi_noun": pmi_noun,
            "pmi_verb": pmi_verb,
            "mention_count": len(nomination_tokens) + len(pronoun_tokens)
        })

    return pd.DataFrame(actor_rows), contains_gnl, generic_masc


# === general helper fuctions ===

def generate_actor_ids(actor_dict):
    """
    Assigns a unique identifier to each actor name.

    Args:
        actor_dict (dict): Dictionary where keys are actor names and values are token lists or similar metadata.

    Returns:
        dict: Mapping from actor names to unique IDs (e.g. "actor_1", "actor_2", ...).
    """
    return {name: f"actor_{i+1}" for i, name in enumerate(actor_dict)}
    
# === nomination ===

def get_actors(text):
    """
    Extract named entities and dependency-based actor mentions from the text.

    This function identifies:
    - Named entities labeled as PERSON (excluding those with apostrophes).
    - Compound modifiers of PERSON entities to capture multi-token names.

    Args:
        text (spacy.tokens.Doc): A SpaCy-parsed document.

    Returns:
        dict: Dictionary mapping actor names (str) to lists of associated tokens.
              Example: {"Angela Merkel": [Token('Angela'), Token('Merkel')]}
    """
    actor_dict = {}

    for token in text:
        # Add compound entities linked to a PERSON
        if token.dep_ == "compound" and token.head.ent_type_ == "PER":
            actor_dict.setdefault(token.head.text, []).append(token)

        # Add individual PERSON entities
        elif token.ent_type_ == "PER" and "'" not in token.text:
            actor_dict.setdefault(token.text, []).append(token)

    return actor_dict


def get_generic_names(text, actor_dict):
    """
    Add generic actor terms (e.g., "frau", "herr") to the actor dictionary.

    This function scans the input text and adds any tokens that match known 
    generic actor terms (defined in `generic_terms`) to the existing actor dictionary.

    Args:
        text (spacy.tokens.Doc): The SpaCy-parsed document.
        actor_dict (dict): Existing dictionary of actors, typically from `get_actors()`.

    Returns:
        dict: Updated actor dictionary including generic names mapped to token lists.
              Example: {"frau": [Token('Frau')], "herr": [Token('Herr')]}
    """

    for token in text:
        token_lower = token.text.lower()
        if token_lower in generic_terms:
            actor_dict.setdefault(token_lower, []).append(token)

    return actor_dict


def combine_names(actor_dict):
    """
    Combine similar or nested actor names by merging their associated tokens.

    This function identifies actor names that are substrings of other names
    (e.g., "Herr Müller" and "Müller") and merges their token lists under the
    more specific name. The shorter or overlapping name is removed afterward.

    Args:
        actor_dict (dict): Dictionary mapping actor names to lists of tokens.

    Returns:
        dict: Updated dictionary with merged actor entries and no redundant keys.
    """
    flagged_keys = {key for key in actor_dict if
                    any(key in second_key for second_key in actor_dict if key != second_key)}
    for key in flagged_keys:
        for second_key in actor_dict:
            if key in second_key:
                actor_dict[second_key].extend(actor_dict[key])
    for key in flagged_keys:
        actor_dict.pop(key, None)

    return actor_dict


def detect_pronouns(text, actor_dict):
    """
    Detects pronoun references to detected actors using coreference resolution.

    This function uses SpaCy's coreferee extension to link pronouns to named actors
    based on resolved coreference chains. Only unambiguous chains (exactly one match)
    are considered. The function builds a mapping from actor names to the list of
    pronouns referring to them.

    Args:
        text (spacy.tokens.Doc): The processed SpaCy document with coreference resolution.
        actor_dict (dict): A dictionary of detected actors and their tokens.

    Returns:
        dict: A dictionary mapping actor names to lists of pronoun tokens that refer to them.
    """
    
    resolved = {k: [] for k in actor_dict}
    for token in text:
        if token.pos_ == "PRON" and hasattr(text._, "coref_chains"):
            resolved_chain = text._.coref_chains.resolve(token)
            if resolved_chain and len(resolved_chain) == 1:
                actor_name = resolved_chain[0].text
                if actor_name in resolved:
                    resolved[actor_name].append(token)
    return resolved


def detect_pronoun_group(pronouns):
    """
    Heuristically determines the dominant pronoun group for an actor.

    This function analyses a list of pronouns associated with an actor and assigns
    a gendered group label based on the proportion of feminine vs. masculine forms.
    If at least 70 percent of pronouns fall into one category, that group is assigned.

    Args:
        pronouns (list[str]): List of pronoun strings assumed to be coreferenced to an actor.

    Returns:
        str: One of the following labels:
             - "she_her" if at least 70% of pronouns are feminine-coded
             - "he_him" if at least 70% are masculine-coded
             - "" (empty string) if no clear majority or list is empty
    """
    if not pronouns:
        return ""
        
    she = {"sie", "ihr", "ihre", "ihren", "ihrem", "ihres"}
    he = {"er", "sein", "seine", "seinen", "seinem", "seines"}
    
    pronouns_lower = [p.lower() for p in pronouns]
    she_count = sum(1 for p in pronouns_lower if p in she)
    he_count = sum(1 for p in pronouns_lower if p in he)
    total = she_count + he_count

    if total == 0:
        return ""
        
    if she_count / total >= 0.7:
        return "she_her"
    elif he_count / total >= 0.7:
        return "he_him"
    else:    
        return ""
    
# === predication ===

def extract_predications(actor_id, actor_name, nomination_tokens, pronoun_tokens):
    """
    Extract sentence-level predications for a given actor, including role, quote type, and frequency.

    For each unique sentence that includes a reference to the actor (either by name or coreferent pronoun),
    this function creates one predication entry annotated with:
    - Grammatical role of the actor (subject, object, or context)
    - Quote type (direct or indirect speech)
    - Number of actor mentions in the sentence

    Heuristics:
    - Grammatical roles are determined from dependency labels.
    - Direct quotes are inferred from the presence of paired quote characters.
    - Indirect quotes are inferred from verbs like "sagen", "erklären", etc.

    Args:
        actor_id (str): Unique ID assigned to the actor (e.g. "actor_1").
        actor_name (str): Canonical name or label for the actor.
        nomination_tokens (list): Tokens where the actor is explicitly named.
        pronoun_tokens (list): Tokens where the actor is referenced via pronouns.

    Returns:
        list[dict]: One dictionary per sentence with the following keys:
            - actor_id
            - actor_name
            - nomination (list of surface forms)
            - pronouns (list of surface forms)
            - predication_text (full sentence)
            - uttered_by_person (bool: is direct quote)
            - is_indirect_quote (bool)
            - syntactic_role (str: "nsubj", "obj", or "context")
            - mention_count (int: number of references to the actor in the sentence)
    """

    SUBJECT_DEPS = {"sb"}
    OBJECT_DEPS = {"oa"}
    
    predication_rows = []
    seen_sentences = set()

    actor_spans = [token.doc[token.i : token.i + 1] for token in nomination_tokens + pronoun_tokens]

    for span in actor_spans:
        sent = span.sent
        if sent in seen_sentences:
            continue
        seen_sentences.add(sent)

        mention_count = sum(1 for tok in sent if any(tok in s for s in actor_spans))
        syntactic_roles = set()
        root = span.root

        if root.dep_ in SUBJECT_DEPS:
            syntactic_roles.add("nsubj")
        elif root.dep_ in OBJECT_DEPS:
            syntactic_roles.add("obj")

        syntactic_role = sorted(syntactic_roles)[0] if syntactic_roles else "context"
        
        # Check for direct quotes
        quote_chars = ['"', '„', '“', '”']
        quote_count = sum(sent.text.count(q) for q in quote_chars)
        contains_direct_quote = quote_count >= 2
        uttered_by_person = contains_direct_quote
        
        # Check for indirect quotes
        verb_lemmas = {token.lemma_ for token in sent if token.pos_ == "VERB"}
        contains_indirect_quote = not contains_direct_quote and bool(indirect_quote_verbs & verb_lemmas)

        predication_rows.append({
            "actor_id": actor_id,
            "actor_name": actor_name,
            "nomination": list(set(t.text for t in nomination_tokens)),
            "pronouns": list(set(t.text for t in pronoun_tokens)),
            "predication_text": sent.text,
            "uttered_by_person": uttered_by_person,
            "is_indirect_quote": contains_indirect_quote,
            "syntactic_role": syntactic_role,
            "mention_count": mention_count
        })

    return predication_rows


# === sentiment ===

def detect_sentiment_score(text):
    """
    Computes a sentiment score for the given text using a preconfigured sentiment analysis pipeline.

    The score is derived from the predicted sentiment label:
    - Positive sentiment returns a positive score (0.0 to 1.0)
    - Negative sentiment returns a negative score (-1.0 to 0.0)
    - Neutral or unclassified returns 0.0

    Args:
        text (str): The sentence or phrase to analyse for sentiment.

    Returns:
        float: Sentiment score in the range [-1.0, 1.0], where negative values indicate negative sentiment
               and positive values indicate positive sentiment.
    """
    try:
        result = sentiment_pipeline(text)
        return (
            result[0]["score"] if result[0]["label"] == "positive" else
            -result[0]["score"] if result[0]["label"] == "negative" else
            0.0
        )
    except Exception:
        return 0.0

# === PMI ===

def compute_pmi(tokens, doc, stopwords, pos="ADJ"):
    """
    Calculates pointwise mutual information (PMI) scores for words that co-occur
    with a specific actor’s references (nomination + attributed pronouns).

    Each word’s PMI reflects its association with sentences that mention the actor.

    Args:
        tokens (list[Token]): All nomination and pronoun tokens referring to the actor.
        doc (spacy.tokens.Doc): Full parsed SpaCy document.
        stopwords (set): Set of stopwords to exclude from counts.
        pos (str): POS tag to restrict analysis (default: "ADJ").

    Returns:
        dict: {word: PMI score}, where higher values mean stronger association
              between the word and the actor’s references.
    """
    actor_sentences = set(t.sent for t in tokens)

    actor_word_freq = Counter()
    doc_word_freq = Counter()
    total_actor_words = 0
    total_doc_words = 0

    for tok in doc:
        if tok.pos_ != pos or not tok.is_alpha:
            continue
        word = tok.text.lower()
        if word in stopwords:
            continue

        doc_word_freq[word] += 1
        total_doc_words += 1

        if tok.sent in actor_sentences:
            actor_word_freq[word] += 1
            total_actor_words += 1

    if total_actor_words == 0 or total_doc_words == 0:
        return {}

    scores = {}
    for word, freq in actor_word_freq.items():
        p_word_given_actor = freq / total_actor_words
        p_word_in_doc = doc_word_freq[word] / total_doc_words
        scores[word] = math.log2(p_word_given_actor / p_word_in_doc)

    return scores



# === gender neutral language ===

def detect_gender_neutral_language(doc):
    """
    Heuristically detects the use of gender-neutral language in a German-language text.

    The method searches for gender-neutral markers such as Binnen-I (e.g. "StudentInnen"),
    colons (e.g. "Student:innen"), asterisks ("Student*innen"), and underscores ("Student_innen"),
    typically used to include all gender identities. It focuses on capitalised common nouns 
    (likely referring to people) that are not part of named entities.

    Args:
        doc (spacy.tokens.Doc): The processed SpaCy document to analyse.

    Returns:
        bool: True if gender-neutral language is used in a meaningful way (≥ 3 occurrences),
              otherwise False.
    """
    # Pattern for Binnen-I, :, *, _ forms — approximates gender-neutral spellings
    gender_neutral_pattern = re.compile(r"\b\w*(?:[:*/_]|[iI]n{2})(?:nen)?\b")

    # Filter to likely genderable common nouns: capitalised, NOUN, not named entities
    candidate_words = [
        token.text
        for token in doc
        if token.pos_ == "NOUN"
        and token.text.istitle()
        and not token.ent_type_  # skip named entities like "Berlin"
        and len(token.text) > 4
    ]

    if not candidate_words:
        return False

    # Count gender-neutral patterns
    gender_neutral_count = sum(
        1 for word in candidate_words if gender_neutral_pattern.search(word)
    )

    # Meaningful usage threshold: at least 3 uses
    return gender_neutral_count >= 3 

    
    
def detect_generic_masculine(doc):
    """
    Heuristically detects the use of generic masculine language in a German-language text.

    The function assumes generic masculine usage if the text contains capitalised common nouns
    (likely referring to people) but *no* occurrences of gender-neutral spellings such as
    Binnen-I (e.g. "StudentInnen"), colons (e.g. "Student:innen"), asterisks ("Student*innen"),
    underscores ("Student_innen"), or slashes ("Student/-innen").

    Args:
        doc (spacy.tokens.Doc): The processed SpaCy document to analyse.

    Returns:
        bool: True if gendered common nouns are used and none follow gender-neutral conventions,
              indicating possible generic masculine usage. False otherwise.
    """
    # Pattern for Binnen-I, :, *, _, / forms — approximates gender-neutral spellings
    gender_neutral_pattern = re.compile(r"\b\w*(?:[:*/_]|[iI]n{2})(?:nen)?\b")

    # Filter to likely genderable common nouns: capitalised, NOUN, not named entities
    candidate_words = [
        token.text
        for token in doc
        if token.pos_ == "NOUN"
        and token.text.istitle()
        and not token.ent_type_  # skip named entities like "Berlin"
        and len(token.text) > 4
    ]

    if not candidate_words:
        return False

    # Count gender-neutral patterns
    gender_neutral_count = sum(
        1 for word in candidate_words if gender_neutral_pattern.search(word)
    )

    # If there are no gender-neutral words but there are gendered candidates, it’s generic masculine
    return gender_neutral_count == 0 and len(candidate_words) > 0


def count_gendered_stems(text, gendered_stems):
    """
    Counts occurrences of gender-coded stems in a given text.

    The function checks each word in the input text (lowercased) and increments
    a counter whenever a word contains any of the specified stems as substrings.
    Useful for detecting gendered language patterns (e.g., "förder", "team", etc.).

    Args:
        text (str): The sentence or text span to analyse.
        gendered_stems (set): A set of lowercase substrings representing gender-coded word stems.

    Returns:
        int: Number of times a gendered stem appears in any word in the input text.
    """
    words = text.lower().split()
    return sum(1 for word in words for stem in gendered_stems if stem in word)



