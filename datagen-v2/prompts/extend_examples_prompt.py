from typing import List
from pydantic import BaseModel, Field

system_prompt = """## You are an expert in data labeling and language annotation. Your job is to generate clear and diverse tag examples.
When a tag has minimal or vague examples, you should generate a more complete examples using the user's initial tag’s name, desc and examples.

Your expanded examples should include:
### A variety of formats and contexts where the tag applies
### Different linguistic styles (formal, informal, technical, etc.)

Keep the tone concise, objective, and suitable for use in a tagging guideline or labeling system.
"""

user_prompt = """Please generate {{ k }} **descriptive examples** for the following tag:
**Tag Name:** {{ tag.name }}
**Description:** {{ tag.desc }}
**Basic examples:** {{ tag.examples | random_sample(3) | join(', ') }}"""

response_format_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "ExtendedTagExamples",
        "description": "A JSON schema for generating extended examples of a tag.",
        "schema": {
            "type": "object",
            "properties": {
                "extended_examples": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "An extended example of the tag",
                    },
                }
            },
            "required": ["extended_examples"],
        },
    },
}


class ExtendedTagExamples(BaseModel):
    extended_examples: List[str] = Field(
        ..., description="A list of extended examples for the tag"
    )
