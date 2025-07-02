package prompts

import (
	"html/template"
	"strings"
)

// ExtendDescriptionSystem frames the LLM for description extension.
const ExtendDescriptionSystem = `## You are an expert in data labeling and language annotation. Your job is to help clarify and enrich tag definitions.
When a tag has minimal or vague description, you should generate a more complete and informative one using the tag’s name and examples. Make sure that the description is within 100 words.

Your expanded description should explain:
### What the tag represents
### When it is typically used
### Any patterns or rules that help identify when the tag applies

Keep the tone concise, objective, and suitable for use in a tagging guideline or labeling system.
`

// ExtendDescriptionUser is the user‐role template.
const ExtendDescriptionUser = `Please generate a **clarified description** for the following tag:
**Tag Name:** {{ .Tag.Name }}
**Basic Description:** {{ .Tag.Desc }}
**Examples:** {{ join (randomSample .Tag.Examples 3) ", " }}

The output should be a paragraph that expands on the original description and makes it clearer when and how this tag should be applied in real-world text.
`

// ExtendDescriptionFormat is the JSON‐schema directive to the API.
var ExtendDescriptionFormat = map[string]interface{}{
	"type": "json_schema",
	"json_schema": map[string]interface{}{
		"name":        "ExtendedTagDescription",
		"description": "A JSON schema for generating an extended description of a tag.",
		"schema": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"extended_description": map[string]interface{}{
					"type":        "string",
					"description": "An extended and clarified description of the tag",
				},
			},
			"required": []string{"extended_description"},
		},
	},
}

// ExtendDescriptionTmpl is the compiled user template.
var ExtendDescriptionTmpl = template.Must(template.New("extendDescription").
	Funcs(template.FuncMap{
		"randomSample": RandomSample,
		"join":         strings.Join,
	}).
	Parse(ExtendDescriptionUser))
