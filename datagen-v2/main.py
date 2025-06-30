import pandas as pd
from data_factory import DataFactory

tags = {
    "ADDRESS": {
        "desc": "A full or partial physical address including street, city, state, or ZIP code.",
        "examples": ["123 Main St", "Los Angeles, CA", "75001"],
    },
    "COMPANY": {
        "desc": "Name of a business or organization.",
        "examples": ["Google", "Acme Corp", "United Nations"],
    },
    "DATE": {
        "desc": "Calendar dates in various formats.",
        "examples": ["January 1, 2023", "01/01/2023", "2023-01-01"],
    },
    "EMAIL": {
        "desc": "Email addresses.",
        "examples": ["john.doe@example.com", "info@company.org"],
    },
    "ETHNICITY": {
        "desc": "Mentions of ethnic or racial identity.",
        "examples": ["Asian", "Hispanic", "Caucasian"],
    },
    "GENDER": {
        "desc": "Mentions of gender identity.",
        "examples": ["male", "female", "non-binary"],
    },
    "ID_NUMBER": {
        "desc": "Identifiers such as Social Security Numbers or passport numbers.",
        "examples": ["123-45-6789", "A1234567", "DL#987654321"],
    },
    "LOCATION": {
        "desc": "Geographical names like cities, countries, or landmarks.",
        "examples": ["Paris", "Mount Everest", "New York"],
    },
    "NAME": {
        "desc": "First names, last names, or full names of individuals.",
        "examples": ["John", "Maria Gomez", "Dr. Smith"],
    },
    "PHONENUMBER": {
        "desc": "Phone numbers in any common format.",
        "examples": ["(123) 456-7890", "123-456-7890", "+1-800-555-1234"],
    },
    "SEXUAL_ORIENTATION": {
        "desc": "Mentions of sexual orientation.",
        "examples": ["gay", "lesbian", "bisexual"],
    },
    "URL": {
        "desc": "Website URLs.",
        "examples": ["https://example.com", "www.openai.com"],
    },
}

tag_info = [
    {
        "name": tag.upper(),
        "desc": info["desc"],
        "examples": info["examples"],
    }
    for tag, info in tags.items()
]


if __name__ == "__main__":
    current_time = pd.Timestamp.now().strftime("%H-%M-%S")
    factory = DataFactory(
        out_dir=f"generated_data/{current_time}",
        openai_key="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",
    )

    factory.generate(tags_info=tag_info, k=100)
