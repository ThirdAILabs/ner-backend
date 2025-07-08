package prompts

import (
	"html/template"
	"strings"
)

// ContextSystem frames the LLM for generating tag contexts.
const ContextSystem = `##You are a specialist in linguistic context modeling and applied taxonomy design.
## Task
Assist user understand the appropriate contexts where a given tag should be applied. Based on the tag's name, description, and examples, generate a diverse and realistic set of **brief scenario labels**—such as types of text, formats, or communication settings—where this tag is commonly used. Each output should be concise, ideally just a few words (e.g., “Conversation transcripts”, “Social media posts”).`

// ContextUser is the user‐role template.
const ContextUser = `Here is the tag information:
**Tag Name:** {{ .Tag.Name }}
**Description:** {{ .Tag.Desc }}
**Examples:** {{ join (randomSample .Tag.Examples 10) ", " }}

Please generate {{ .K }} short scenario labels (just a few words each) where this tag would appropriately apply. Each label should describe a type of text, document, or communication context where {{ .Tag.Name }} would typically appear.
`

// ContextFormat is the JSON‐schema directive to the API.
var ContextFormat = map[string]interface{}{
	"type": "json_schema",
	"json_schema": map[string]interface{}{
		"name":        "TagContextScenarios",
		"description": "A JSON schema for generating scenarios where a tag can be appropriately used.",
		"schema": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"scenarios": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "A short, distinct textual space or scenario where the tag is relevant.",
				},
			},
			"required": []string{"scenarios"},
		},
	},
}

// ContextTmpl is the compiled user template.
var ContextTmpl = template.Must(template.New("tagContext").
	Funcs(template.FuncMap{
		"randomSample": RandomSample,
		"join":         strings.Join,
	}).
	Parse(ContextUser))
