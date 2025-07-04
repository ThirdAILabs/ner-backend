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
{%- if tag.examples | length > 0 %}
**Tag examples:** {{ tag.examples | random_sample(10) | join(', ') }}
{%- endif %}
{%- if not feedback and (tag.context | length > 0) -%}
**Tag context:** {{ tag.context | random_sample(4) | join(', ') }}
{%- endif %}
{% endfor %}

> Token-Tag pair should be enclosed in this tagging format : ##tokens##TAG##
{% if feedback -%}
# Below are some Contextual examples.
{%- for example in feedback | random_sample(4) %}
- {{- example }}
{% endfor %}
{% else %}
- "##Karun naiyar##NAME## lives at ##123 Main St, Springfield##ADDRESS##, and his birthday is ##January 1, 1990##DATE##."
- "Working at Acme Corp, located in ##Los Angeles##ADDRESS##, and ##Maria Gomez##NAME## was born on ##March 15, 1985##DATE##."
{% endif -%}

## Requirements:
{%- for req in requirements %}
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

    def clean(self):
        """
        Cleans the sentences by removing any leading or trailing whitespace.
        """
        self.sentences = [
            sentence.strip() for sentence in self.sentences if sentence.strip()
        ]
        return self
