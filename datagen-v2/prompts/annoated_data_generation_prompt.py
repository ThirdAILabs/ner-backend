from typing import List
from pydantic import BaseModel, Field

system_prompt = """# You are a data generation assistant specialized in creating high-quality, diverse, and realistic text samples for token classification tasks.
## Goal:
Your goal is to generate natural sentences where multiple types of entities (tags) appear in varied and realistic linguistic contexts. 
"""

user_prompt = """## Task: Generate {{ k }} diverse and realistic sentences containing tokens labeled with the following tags.

### Tags information:
{%- for tag in tagInfo %}
**Tag name:** {{ tag.name }}
**Tag description:** {{ tag.desc }}
**Tag examples:** {{ tag.examples | random_sample(10) | join(', ') }}
**Tag contexts:** {{ tag.contexts | random_sample(10) | join(', ') }}
{% endfor %}

> Tagging format: ##entity text##TAG##
> Example sentences for tags NAME, ADDRESS, and DATE:
1. "##Karun naiyar##NAME## lives at ##123 Main St, Springfield##ADDRESS##, and his birthday is ##January 1, 1990##DATE##."
2. "Working at Acme Corp, located in ##Los Angeles##ADDRESS##, and ##Maria Gomez##NAME## was born on ##March 15, 1985##DATE##."

### Requirements:
- Use varying sentence lengths: short (2–10 words), medium (10–30 words), and long (30+ words, preferred).
- Preferrably try to include many tags in each sentence, but also allow for less-tag sentences.
- Include data from multiple contexts as specified in the tag information but not limited to those contexts.
- Where appropriate, simulate misspellings, slang, or typos.
{%- for req in requirements | random_sample(5) %}
- {{ req -}}
{% endfor %}

{% if user_instructions -%}
### Additional user instructions:
{%- for req in user_instructions %}
- {{ req -}}
{% endfor -%}
{% endif -%}"""

response_format_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "AnnotatedData",
        "description": "A set of sentences with tokens labeled with specified tags for token classification tasks.",
        "schema": {
            "type": "object",
            "properties": {
                "sentences": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A sentence containing tokens labeled with specified tags.",
                    },
                }
            },
            "required": ["sentences"],
        },
    },
}


class AnnotatedData(BaseModel):
    sentences: List[str] = Field(
        ...,
        description="A set of sentences with tokens labeled with specified tags for token classification tasks.",
    )
