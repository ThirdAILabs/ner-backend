from typing import List
from pydantic import BaseModel, Field

system_prompt = """## You are an expert in data labeling and language annotation. Your job is to help clarify and enrich tag definitions.
When a tag has minimal or vague description, you should generate a more complete and informative one using the tag’s name and examples. Make sure that the description is within 100 words 

Your expanded description should explain:
### What the tag represents
### When it is typically used
### Any patterns or rules that help identify when the tag applies

Keep the tone concise, objective, and suitable for use in a tagging guideline or labeling system.
"""

user_prompt = """Please generate a **clarified description** for the following tag:
**Tag Name:** {{ tag.name }}
**Basic Description:** {{ tag.desc }}
**Examples:** {{ tag.examples | random_sample(3) | join(', ') }}
The output should be a paragraph that expands on the original description and makes it clearer when and how this tag should be applied in real-world text.
"""

response_format_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "ExtendedTagDescription",
        "description": "A JSON schema for generating an extended description of a tag.",
        "schema": {
            "type": "object",
            "properties": {
                "extended_description": {
                    "type": "string",
                    "description": "An extended and clarified description of the tag",
                }
            },
            "required": ["extended_description"],
        },
    },
}


class ExtendedTagDescription(BaseModel):
    extended_description: str = Field(
        ..., description="An extended and clarified description of the tag"
    )
