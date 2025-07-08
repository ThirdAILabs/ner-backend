package prompts

import (
	"html/template"
	"strings"
)

const ExtendExamplesSystem = `## You are a specialist in applied linguistics and example curation for annotation systems.
## Task
Generate a diverse and representative examples of the tag using the tag’s name, description, and any initial examples provided.
Your examples should include:
- A range of formats and communication contexts where the tag applies
- Variation in linguistic style (e.g., formal, informal, technical, conversational)
Keep the tone concise, objective, and suitable for use in a tagging guideline or labeling system.
`

const ExtendExamplesUser = `Please generate {{ .K }} **diverse examples** for the following tag:
**Tag Name:** {{ .Tag.Name }}
**Description:** {{ .Tag.Desc }}
**Basic examples:** {{ join (randomSample .Tag.Examples 3) ", " }}
`

var ExtendExamplesFormat = map[string]interface{}{
	"type": "json_schema",
	"json_schema": map[string]interface{}{
		"name":        "ExtendedTagExamples",
		"description": "A JSON schema for generating extended examples of a tag.",
		"schema": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"extended_examples": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "An extended example of the tag",
				},
			},
			"required": []string{"extended_examples"},
		},
	},
}

var ExtendExamplesTmpl = template.Must(template.New("extendExamples").
	Funcs(template.FuncMap{
		"randomSample": RandomSample,
		"join":         strings.Join,
	}).
	Parse(ExtendExamplesUser))
