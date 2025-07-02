package prompts

import (
	"html/template"
	"strings"
)

// ExtendExamplesSystem frames the LLM for example extension.
const ExtendExamplesSystem = `## You are an expert in data labeling and language annotation. Your job is to generate clear and diverse tag examples.
When a tag has minimal or vague examples, you should generate more complete examples using the tag’s name, desc, and initial examples.

Your expanded examples should include:
### A variety of formats and contexts where the tag applies
### Different linguistic styles (formal, informal, technical, etc.)

Keep the tone concise, objective, and suitable for use in a tagging guideline or labeling system.
`

// ExtendExamplesUser is the user‐role template.
const ExtendExamplesUser = `Please generate {{ .K }} **descriptive examples** for the following tag:
**Tag Name:** {{ .Tag.Name }}
**Description:** {{ .Tag.Desc }}
**Basic examples:** {{ join (randomSample .Tag.Examples 3) ", " }}
`

// ExtendExamplesFormat is the JSON‐schema directive to the API.
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

// ExtendExamplesTmpl is the compiled user template.
var ExtendExamplesTmpl = template.Must(template.New("extendExamples").
	Funcs(template.FuncMap{
		"randomSample": RandomSample,
		"join":         strings.Join,
	}).
	Parse(ExtendExamplesUser))
