def build_tag_vocab():
    tag_set = set()
    tags = [
        "ADDRESS",
        "CARD_NUMBER",
        "COMPANY",
        "CREDIT_SCORE",
        "DATE",
        "EMAIL",
        "ETHNICITY",
        "GENDER",
        "ID_NUMBER",
        "LICENSE_PLATE",
        "LOCATION",
        "NAME",
        "PHONENUMBER",
        "SERVICE_CODE",
        "SEXUAL_ORIENTATION",
        "SSN",
        "URL",
        "VIN",
        "O",
    ]

    for tag in tags:
        tag_set.add(tag)

    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}
    return tag2idx
