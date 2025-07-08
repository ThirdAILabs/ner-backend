package prompts

import (
	"html/template"
	"strings"
)

// ExtendDescriptionSystem frames the LLM for description extension.
const ExtendDescriptionSystem = `You are a specialist in semantic definition writing and taxonomy development.
## Task
Generate a clearer and more informative version of the tag description using the it's name and examples. Keep the description under 60 words.
## Your expanded description should explain:
- **What** the tag represents
- **When** it is typically applied
- **How** to recognize it (common patterns, signals, or rules)
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
