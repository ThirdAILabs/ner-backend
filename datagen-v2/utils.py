import os
import csv
import random
from typing import List, Dict, Optional

requirements = [
    "Some sentences must be grammatically correct and naturally written.",
    "Avoid robotic or templated phrasing; all sentences should feel human-authored.",
    "Ensure fluent sentence structure with realistic syntax and semantics.",
    "Include both formal and informal sentence styles."
    "Maintain linguistic variety: mix sentence types (declarative, interrogative, exclamatory).",
    "Avoid repetition of sentence patterns, phrases, and vocabulary.",
    "Sentences can be from imaginary or real-world contexts.",
    "Use contextually rich phrases that give meaning to the tagged entities.",
    "Ensure each tag appears in at least X different sentence contexts.",
    "Cover a wide range of surface forms for each tag",
    "Include variations in capitalization and punctuation where natural.",
    "Do not use the same example/entity more than once per tag unless rephrased significantly.",
    "Simulate conversations, tweets, formal documents, blog posts, and questions.",
    "Vary tone: use professional, casual, emotional, sarcastic, humorous tones across samples.",
    "Use a wide variety of vocabulary, idioms, and expressions.",
    "Incorporate some sentences from different regions or dialects (e.g., British vs. American English).",
    'All named entities should be plausible and realistic (e.g., "John Smith" not "Name Placeholder").'
    "Avoid overuse of rare or uncommon formats unless explicitly requested.",
    "Ensure even distribution of tag types across the dataset.",
    "Provide balanced representation of gender, ethnicity, region, and culture in names and contexts.",
    "Simulate real-world noise like: Spelling mistakes, Missing punctuation, Abbreviations",
    'Insert some occasional irrelevant tokens or fillers (e.g., "umm", "you know", "well").',
    "Mimic automatic speech transcription quirks in a subset of examples.",
    "Include some sentences where entity boundaries are ambiguous.",
    "Include some nested entities or overlapping meanings if needed for advanced models.",
    'Include some examples with acronyms or abbreviations for named entities (e.g., "IBM", "WHO").',
    "Include some polysemous words as entities or non-entities (e.g., “Apple” as ORG vs fruit)",
    "Include some confusable formats (e.g., phone number vs ID number vs money).",
    'Include some rare or less common entity examples (e.g., "##Yamoussoukro$$LOCATION##").',
    "Include some adversarial examples where the entity could be mistaken for a different tag.",
    "Add some ungrammatical or ill-formed but understandable constructions (e.g., tweets or SMS).",
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
