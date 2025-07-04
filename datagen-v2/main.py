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
    # "DATE": {
    #     "desc": "Calendar dates in various formats.",
    #     "examples": ["January 1, 2023", "01/01/2023", "2023-01-01"],
    # },
    # "EMAIL": {
    #     "desc": "Email addresses.",
    #     "examples": ["john.doe@example.com", "info@company.org"],
    # },
    # "ETHNICITY": {
    #     "desc": "Mentions of ethnic or racial identity.",
    #     "examples": ["Asian", "Hispanic", "Caucasian"],
    # },
    # "GENDER": {
    #     "desc": "Mentions of gender identity.",
    #     "examples": ["male", "female", "non-binary"],
    # },
    # "ID_NUMBER": {
    #     "desc": "Identifiers such as Social Security Numbers or passport numbers.",
    #     "examples": ["123-45-6789", "A1234567", "DL#987654321"],
    # },
    # "LOCATION": {
    #     "desc": "Geographical names like cities, countries, or landmarks.",
    #     "examples": ["Paris", "Mount Everest", "New York"],
    # },
    "NAME": {
        "desc": "First names, last names, or full names of individuals.",
        "examples": ["John", "Maria Gomez", "Dr. Smith"],
    },
    "PHONENUMBER": {
        "desc": "Phone numbers in any common format.",
        "examples": ["(123) 456-7890", "123-456-7890", "+1-800-555-1234"],
    },
    # "SEXUAL_ORIENTATION": {
    #     "desc": "Mentions of sexual orientation.",
    #     "examples": ["gay", "lesbian", "bisexual"],
    # },
    # "URL": {
    #     "desc": "Website URLs.",
    #     "examples": ["https://example.com", "www.openai.com"],
    # },
    # "CARD_NUMBER": {
    #     "desc": "Credit or debit card numbers.",
    #     "examples": ["4111 1111 1111 1111", "5500 1234 5678 9010"],
    # },
    # "CREDIT_SCORE": {
    #     "desc": "Credit scores or ratings.",
    #     "examples": ["700", "750", "Excellent"],
    # },
    # "LICENSE_PLATE": {
    #     "desc": "Vehicle license plate numbers.",
    #     "examples": ["ABC1234", "XYZ-5678", "1A2B3C"],
    # },
    # "SERVICE_CODE": {
    #     "desc": "CVV codes or service codes of credit/debit cards.",
    #     "examples": ["123", "456", "789"],
    # },
    # "SSN": {
    #     "desc": "Social Security Numbers of us citizens.",
    #     "examples": ["123-45-6789", "987-65-4320"],
    # },
    # "VIN": {
    #     "desc": "Vehicle Identification Numbers of vehicles.",
    #     "examples": ["1HGCM82633A123456", "1FTRX18W51NA12345"],
    # },
}

tag_info = [
    {
        "name": tag.upper(),
        "desc": info["desc"],
        "examples": info["examples"],
    }
    for tag, info in tags.items()
]
feedback = []

openai_key = ""
if __name__ == "__main__":
    current_time = pd.Timestamp.now().strftime("%H-%M-%S")
    factory = DataFactory(
        out_dir=f"generated_data/{current_time}",
        openai_key=openai_key,
    )

    factory.generate(
        tags_info=tag_info, k=50, generate_per_llm_call=20, write_batch_size=1000
    )
