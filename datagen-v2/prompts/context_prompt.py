from typing import List
from pydantic import BaseModel, Field

system_prompt = """## You are an expert in text classification and annotation. 
## Task
Help users understand the appropriate contexts where a given tag should be applied. Based on the tag's name, description, and examples, generate a diverse and realistic set of **brief scenario labels**—such as types of text, formats, or communication settings—where this tag is commonly used. Each output should be concise, ideally just a few words (e.g., “Conversation transcripts”, “Social media posts”)."""

user_prompt = """Here is the tag information:
**Tag Name:** {{ tag.name }}
**Description:** {{ tag.desc }}
**Examples:** {{ tag.examples | random_sample(10) | join(', ') }}
Please generate {{ k }} short scenario labels (just a few words each) where this tag would appropriately apply. Each label should describe a type of text, document, or communication context where {{ tag.name }} would typically appear."""

response_format_json = {
    "type": "json_schema",
    "json_schema": {
        "name": "TagContextScenarios",
        "description": "A JSON schema for generating scenarios where a tag can be appropriately used.",
        "schema": {
            "type": "object",
            "properties": {
                "scenarios": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A short, distinct textual space or scenario where the tag is relevant.",
                    },
                }
            },
            "required": ["scenarios"],
        },
    },
}


class TagContextScenarios(BaseModel):
    scenarios: List[str] = Field(
        ...,
        description="A list of distinct textual spaces or scenarios where the tag is relevant.",
    )
