from typing import List, Dict
from pydantic import BaseModel, Field

system_prompt = """# You are an expert NER tag correction assistant.
## Objective:
Your task is to review and correct entity tagging in annotated text samples used for token classification tasks. Ensure that all named entity recognition (NER) tags are accurate, consistent, and conform to the expected labeling schema.

## Tagging Format:
All entity mentions must follow this exact format:
**##entity text##ENTITY_TYPE##**

## Guidelines:
1. **Preserve correctly tagged entities.** Do not alter or remove correct tags.
2. **Fix incorrect or missing tags.** If an entity is mislabeled or untagged, correct it using the proper format.
3. **Remove inappropriate tags.** If a tag is applied to non-entity text, remove it.
4. **Ensure consistency and accuracy** with the labeling schema and context.

### Example 1:
Incorrectly annotated text:
```
Emma watson is a software engineer at ##TechCorp##COMPANY. He lives in ##San Francisco##LOCATION.
```
Corrected annotated text:
```
##Emma watson##NAME## is a software engineer at ##TechCorp##COMPANY##. He lives in ##San Francisco##LOCATION##.
```
Corrections of the above example:
- 'Emma watson' should be tagged as "##Emma watson##NAME##".
- The tag LOCATION was incorrectly applied to the token "San Francisco". It should be corrected to "##San Francisco##LOCATION##".
---

### Example 2:
Incorrectly annotated text:
```
##Jane Smith##NAME is a ##data scientist## at DataSolutions Inc. She works remotely from ##New York##LOCATION## and is reachable at ##jane.smith@example.com##PHONE_NUMBER##.
```
Corrected annotated text:
```
##Jane Smith##NAME## is a data scientist at ##DataSolutions Inc.##COMPANY##. She works remotely from ##New York##LOCATION## and is reachable at ##jane.smith@example.com##EMAIL##.
```
Corrections of the above example:
- The tag COMPANY was missed for the token "DataSolutions Inc.". It should be corrected to "##DataSolutions Inc.##COMPANY##".
- No tag for data scientist and if no tag is applicable, it should be removed.
- The tag PHONE_NUMBER was incorrectly applied to the token 'jane.smith@example.com'. It should be corrected to "##jane.smith@example.com##EMAIL##".
---

### Example 3:
Incorrectly annotated text:
```
##The Red Cross coordinated##Company## their relief efforts from their base in ##Ottawa, Ontario##LOCATION##.
```
Corrected annotated text:
```
##The Red Cross##COMPANY## coordinated their relief efforts from their base in ##Ottawa, Ontario##LOCATION##.
``` 
Corections of the above example:
- The tag Company was incorrectly applied to the token "The Red Cross coordinated". It should be corrected to "##The Red Cross##COMPANY## coordinated".
"""

user_prompt = """Given the following annotated text samples, verify the correctness of the tags and correct them if necessary.
### Tags information:
{%- for tag in tagInfo %}
**Tag name:** {{ tag.name }}
**Tag description:** {{ tag.desc }}
{% endfor %}

Annotated text samples:
{%- for sample in annotated_texts %}
```
{{ sample }}
```
{% endfor %}

Return only the corrected annotated text samples that required corrections, in the same format as the examples provided in the system prompt.
If there are no corrections needed, return empty.
"""

response_format_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "AnnotatedTextSamples",
        "description": "A set of annotated text samples with verified and corrected tags for token classification tasks.",
        "schema": {
            "type": "object",
            "properties": {
                "annotated_texts": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "An annotated text sample with verified and corrected tags.",
                    },
                }
            },
            "required": ["annotated_texts"],
        },
    },
}


class AnnotatedTextSamples(BaseModel):
    annotated_texts: List[str] = Field(
        ...,
        description="A set of annotated text samples with verified and corrected tags for token classification tasks.",
    )

    def clean(self):
        """
        Cleans the annotated texts by removing any leading or trailing whitespace.
        """
        self.annotated_texts = [
            text.strip() for text in self.annotated_texts if text.strip()
        ]
        return self
