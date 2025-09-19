# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import re
import string

ARTICLE_REGEX = re.compile(r"\b(a|an|the|of|is)\b", re.UNICODE)
PUNCTUATION_TRANSLATOR = str.maketrans("", "", string.punctuation)

Y_ENDING = re.compile(r"[^aeiou]y$")
S_ENDING = re.compile(r"[sxz]$")
SH_CH_ENDING = re.compile(r"(sh|ch)$")
IES_ENDING = re.compile(r"ies$")
ES_ENDING = re.compile(r"es$")

SUFFIX_PATTERN = re.compile(r"_\d+$")

IRREGULAR_PLURALS = {
    "addendum": "addenda",
    "aircraft": "aircraft",
    "alga": "algae",
    "alumna": "alumnae",
    "alumnus": "alumni",
    "alveolus": "alveoli",
    "amoeba": "amoebae",
    "analysis": "analyses",
    "antenna": "antennae",
    "antithesis": "antitheses",
    "apex": "apices",
    "appendix": "appendices",
    "automaton": "automata",
    "axis": "axes",
    "bacillus": "bacilli",
    "bacterium": "bacteria",
    "baculum": "bacula",
    "barracks": "barracks",
    "basis": "bases",
    "beau": "beaux",
    "bison": "bison",
    "buffalo": "buffalo",
    "bureau": "bureaus",
    "cactus": "cacti",
    "calf": "calves",
    "carcinoma": "carcinomata",
    "carp": "carp",
    "census": "censuses",
    "chassis": "chassis",
    "cherub": "cherubim",
    "child": "children",
    "château": "châteaus",
    "cloaca": "cloacae",
    "cod": "cod",
    "codex": "codices",
    "concerto": "concerti",
    "consortium": "consortia",
    "corpus": "corpora",
    "crisis": "crises",
    "criterion": "criteria",
    "curriculum": "curricula",
    "cystoma": "cystomata",
    "datum": "data",
    "deer": "deer",
    "diagnosis": "diagnoses",
    "die": "dice",
    "dwarf": "dwarfs",
    "echo": "echoes",
    "elf": "elves",
    "elk": "elk",
    "ellipsis": "ellipses",
    "embargo": "embargoes",
    "emphasis": "emphases",
    "erratum": "errata",
    "faux pas": "faux pas",
    "fez": "fezes",
    "firmware": "firmware",
    "fish": "fish",
    "focus": "foci",
    "foot": "feet",
    "formula": "formulae",
    "fungus": "fungi",
    "gallows": "gallows",
    "genus": "genera",
    "glomerulus": "glomeruli",
    "goose": "geese",
    "graffito": "graffiti",
    "grouse": "grouse",
    "half": "halves",
    "hamulus": "hamuli",
    "hero": "heroes",
    "hippopotamus": "hippopotami",
    "hoof": "hooves",
    "hovercraft": "hovercraft",
    "hypothesis": "hypotheses",
    "iliac": "ilia",
    "incubus": "incubi",
    "index": "indices",
    "interstitium": "interstitia",
    "kakapo": "kakapo",
    "knife": "knives",
    "larva": "larvae",
    "leaf": "leaves",
    "libretto": "libretti",
    "life": "lives",
    "loaf": "loaves",
    "loculus": "loculi",
    "locus": "loci",
    "louse": "lice",
    "man": "men",
    "matrix": "matrices",
    "means": "means",
    "measles": "measles",
    "media": "media",
    "medium": "media",
    "memorandum": "memoranda",
    "millennium": "millennia",
    "minutia": "minutiae",
    "moose": "moose",
    "mouse": "mice",
    "nebula": "nebulae",
    "nemesis": "nemeses",
    "neurosis": "neuroses",
    "news": "news",
    "nucleolus": "nucleoli",
    "nucleus": "nuclei",
    "oasis": "oases",
    "occiput": "occipita",
    "offspring": "offspring",
    "omphalos": "omphaloi",
    "opus": "opera",
    "ovum": "ova",
    "ox": "oxen",
    "paralysis": "paralyses",
    "parenthesis": "parentheses",
    "person": "people",
    "phenomenon": "phenomena",
    "phylum": "phyla",
    "pike": "pike",
    "polyhedron": "polyhedra",
    "potato": "potatoes",
    "primus": "primi",
    "prognosis": "prognoses",
    "quiz": "quizzes",
    "radius": "radii",
    "referendum": "referenda",
    "salmon": "salmon",
    "scarf": "scarves",
    "scrotum": "scrota",
    "self": "selves",
    "seminoma": "seminomata",
    "series": "series",
    "sheep": "sheep",
    "shelf": "shelves",
    "shrimp": "shrimp",
    "simulacrum": "simulacra",
    "soliloquy": "soliloquies",
    "spacecraft": "spacecraft",
    "species": "species",
    "spectrum": "spectra",
    "squid": "squid",
    "stimulus": "stimuli",
    "stratum": "strata",
    "swine": "swine",
    "syconium": "syconia",
    "syllabus": "syllabi",
    "symposium": "symposia",
    "synopsis": "synopses",
    "synthesis": "syntheses",
    "tableau": "tableaus",
    "testis": "testes",
    "that": "those",
    "thesis": "theses",
    "thief": "thieves",
    "this": "these",
    "thrombus": "thrombi",
    "tomato": "tomatoes",
    "tooth": "teeth",
    "torus": "tori",
    "trout": "trout",
    "tuna": "tuna",
    "umbilicus": "umbilici",
    "uterus": "uteri",
    "vertebra": "vertebrae",
    "vertex": "vertices",
    "veto": "vetoes",
    "vita": "vitae",
    "vortex": "vortices",
    "watercraft": "watercraft",
    "wharf": "wharves",
    "wife": "wives",
    "wolf": "wolves",
    "woman": "women",
}

IRREGULAR_SINGULARS = {plural: singular for singular, plural in IRREGULAR_PLURALS.items()}


def to_plural(word):
    """
    Convert a singular word to its plural form.

    Args:
        word (str): The singular word to convert.

    Returns:
        (str): The plural form of the word.
    """
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS.get(word)
    else:
        # Use rules for regular plurals
        if Y_ENDING.search(word):
            return f"{word[:-1]}ies"
        elif S_ENDING.search(word) or SH_CH_ENDING.search(word):
            return f"{word}es"
        else:
            return f"{word}s"


def to_singular(word):
    """
    Convert a plural word to its singular form.

    Args:
        word (str): The plural word to convert.

    Returns:
        (str): The singular form of the word.
    """
    if word in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS.get(word)
    else:
        # Use rules for regular singulars
        if IES_ENDING.search(word):
            return f"{word[:-3]}y"
        elif ES_ENDING.search(word):
            if S_ENDING.search(word[:-2]) or SH_CH_ENDING.search(word[:-2]):
                return word[:-2]
            else:
                return word[:-1]
        elif word.endswith("s"):
            return word[:-1]
        else:
            return word


def to_plural_property(property_key):
    """
    Convert the last word of a property key to its plural form.

    Args:
        property_key (str): The property key to convert.

    Returns:
        (str): The property key with the last word in plural form.
    """
    words = property_key.split("_")
    if len(words) > 1:
        # Assume the last word is the noun
        words[-1] = to_plural(words[-1])
    else:
        words[0] = to_plural(words[0])
    return "_".join(words)


def to_singular_property(property_key):
    """
    Convert the last word of a property key to its singular form.

    Args:
        property_key (str): The property key to convert.

    Returns:
        (str): The property key with the last word in singular form.
    """
    words = property_key.split("_")
    if len(words) > 1:
        # Assume the last word is the noun
        words[-1] = to_singular(words[-1])
    else:
        words[0] = to_singular(words[0])
    return "_".join(words)


def remove_numerical_suffix(property_key):
    """
    Remove the numerical suffix from a property key.

    Args:
        property_key (str): The property key to process.

    Returns:
        (str): The property key with the suffix removed.
    """
    return re.sub(SUFFIX_PATTERN, "", property_key)


def add_suffix(property_key, suffix):
    """
    Add a suffix to a property key.

    Args:
        property_key (str): The property key to process.
        suffix (int): The suffix to add.

    Returns:
        (str): The property key with the suffix added.
    """
    return f"{property_key}_{suffix}"


def to_singular_without_numerical_suffix(property_key):
    """
    Convert a property key to its base (singular) form by removing
        the numerical suffix and converting to singular.

    Args:
        property_key (str): The property key to convert.

    Returns:
        (str): The base (singular) form of the property key.
    """
    property_key = remove_numerical_suffix(property_key)
    return to_singular_property(property_key)


def to_plural_without_numerical_suffix(property_key):
    """
    Convert a property key to its list (plural) form by removing
        the numerical suffix and converting to plural.

    Args:
        property_key (str): The property key to convert.

    Returns:
        (str): The list (plural) form of the property key.
    """
    property_key = remove_numerical_suffix(property_key)
    return to_plural_property(property_key)


def is_plural(property_key):
    """
    Check if the last word of a property key is in plural form.

    Args:
        property_key (str): The property key to check.

    Returns:
        (bool): True if the last word is plural, False otherwise.
    """
    words = property_key.split("_")
    if len(words) > 1:
        noun = words[-1]
    else:
        noun = words[0]

    singular_form = to_singular(noun)
    return singular_form != noun


def remove_articles(text):
    """
    Remove common English articles from the text.

    Args:
        text (str): The text to process.

    Returns:
        (str): The text with articles removed.
    """
    return " ".join(re.sub(ARTICLE_REGEX, "", text).split())


def remove_punctuation(text):
    """
    Remove punctuation from the text.

    Args:
        text (str): The text to process.

    Returns:
        (str): The text with punctuation removed.
    """
    return text.translate(PUNCTUATION_TRANSLATOR)


def normalize_text(text):
    """
    Normalize the text by converting to lowercase, removing articles,
        and removing punctuation.

    Args:
        text (str): The text to normalize.

    Returns:
        (str): The normalized text.
    """
    return remove_articles(remove_punctuation(text.strip().lower()))


def normalize_and_tokenize(text):
    """
    Normalize the text and tokenize it into words.

    Args:
        text (str): The text to process.

    Returns:
        (list): A list of normalized words.
    """
    text = text.lower()
    text = remove_articles(text)
    text = remove_punctuation(text)
    return text.split()


def shorten_text(text, nb_words_offset=10):
    """
    Shorten a text.

    Args:
        text (str): The text to shorten.
        nb_words_offset (int): The number of words to keep
            from the beginning and end of the text
            (Default is 20).

    Returns:
        (str): The shortened text. If the original text has more than nb_words words,
            returns the first nb_words and last nb_words separated by " (...) ".
            Otherwise, returns the original text unchanged.
    """
    if not isinstance(text, str):
        text = str(text)

    words = text.split(" ")
    if len(words) <= nb_words_offset * 2:
        return text
    nb_words_removed = len(words) - 2 * nb_words_offset
    short_text = (
        " ".join(words[:nb_words_offset])
        + f" (... {nb_words_removed} words removed for clarity) "
        + " ".join(words[-nb_words_offset:])
    )
    return short_text
