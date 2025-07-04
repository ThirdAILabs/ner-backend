import os
import csv
import random
from typing import List, Dict, Optional, Set

required_requirements = [
    "Use varying sentence lengths: short (2–10 words), medium (10–30 words), and long (30+ words, preferred).",
    "Preferrably try to include many tags in each sentence, but also allow for less-tag sentences.",
    "Include data from multiple contexts as specified in the tag information but not limited to those contexts.",
    "Where appropriate, simulate misspellings, slang, or typos",
]
additional_requirements = [
    "Some sentences must be grammatically correct and naturally written.",
    "Include fluent sentences structure with realistic syntax and semantics.",
    "Include both formal and informal sentence styles."
    "Maintain linguistic variety: mix sentence types (declarative, interrogative, exclamatory).",
    "Avoid repetition of sentence patterns, phrases, and vocabulary.",
    "Sentences can be from imaginary or real-world contexts.",
    "Use contextually rich phrases that give meaning to the tagged entities.",
    "Ensure each tag appears in at least X different sentence contexts.",
    "Cover a wide range of surface forms for each tag",
    "Do not use the same example/entity more than once per tag unless rephrased significantly.",
    "Simulate conversations, tweets, formal documents, blog posts, and questions.",
    "Vary tone: use professional, casual, emotional, sarcastic, humorous tones across samples.",
    "Use a wide variety of vocabulary, idioms, and expressions.",
    "Incorporate some sentences from different regions or dialects (e.g., British vs. American English).",
    "Provide balanced representation of gender, ethnicity, region, and culture in names and contexts.",
    "Simulate real-world noise like: Spelling mistakes, Missing punctuation, Abbreviations",
    'Insert some occasional irrelevant tokens or fillers (e.g., "umm", "you know", "well").',
    "Mimic automatic speech transcription quirks in a subset of examples.",
    "Include some sentences where entity boundaries are ambiguous.",
    "Include some nested entities or overlapping meanings if needed for advanced models.",
    'Include some examples with acronyms or abbreviations for named entities (e.g., "IBM", "WHO").',
    "Include some polysemous words as entities or non-entities (e.g., “Apple” as ORG vs fruit)",
    "Include some confusable formats (e.g., phone number vs ID number vs money).",
    'Include some rare or less common entity examples (e.g., "Yamoussoukro as a LOCATION").',
    "Include some adversarial examples where the entity could be mistaken for a different tag.",
    "Add some ungrammatical or ill-formed but understandable constructions (e.g., tweets or SMS).",
]

contextual_example_requirements = [
    "Look at the contextual examples above to understand and generate sentences of similar complexity and variety.",
    "Contexual examples could be incorrect in tagging at few places because they are based on human feedback. Try to generate sentences that are accurate and mimics similar to contextual examples.",
    "Make sure not to miss any tag in the generated sentences and ensure that tag is in the correct format as mentioned above.",
    "Also make sure that the sentences are of same length and complexity as the contextual examples.",
]


def write_to_csv(
    path: str,
    data_points: List[Dict[str, str]],
    fieldnames: List[str],
    newline: Optional[str] = None,
    encoding: Optional[str] = None,
):
    if os.path.exists(path):
        mode = "a"
    else:
        mode = "w"

    with open(path, mode, newline=newline, encoding=encoding) as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if mode == "w":
            csv_writer.writeheader()
        csv_writer.writerows(data_points)


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)

    if not union:
        return 0.0
    return len(intersection) / len(union)


def find_most_similar(text: str, sentences: list[str]) -> tuple[str, float]:
    best_sentence = None
    best_score = -1.0

    for sentence in sentences:
        score = jaccard_similarity(text, sentence)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence, best_score
