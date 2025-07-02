package prompts

import (
	"html/template"
	"strings"
)

// AnnotatedDataSystem frames the LLM for annotated data generation.
const AnnotatedDataSystem = `# You are a data generation assistant specialized in creating high-quality, diverse, and realistic text samples for token classification tasks.
## Goal:
Your goal is to generate natural sentences where multiple types of entities (tags) appear in varied and realistic linguistic contexts. 
`

// AnnotatedDataUser is the template for the user message.
const AnnotatedDataUser = `## Task: Generate {{ .K }} diverse and realistic sentences containing tokens labeled with the following tags.

### Tags information:
{{- range .TagInfo }}
**Tag name:** {{ .Name }}
**Tag description:** {{ .Desc }}
**Tag examples:** {{ join (randomSample .Examples 10) ", " }}
**Tag contexts:** {{ join (randomSample .Contexts 10) ", " }}
{{- end }}

> Tagging format: ##entity text##TAG##
> Example sentences for tags NAME, ADDRESS, and DATE:
1. "##Karun naiyar##NAME## lives at ##123 Main St, Springfield##ADDRESS##, and his birthday is ##January 1, 1990##DATE##."
2. "Working at Acme Corp, located in ##Los Angeles##ADDRESS##, and ##Maria Gomez##NAME## was born on ##March 15, 1985##DATE##."

### Requirements:
- Use varying sentence lengths: short (2–10 words), medium (10–30 words), and long (30+ words, preferred).
- Preferably try to include many tags in each sentence, but also allow for less-tag sentences.
- Include data from multiple contexts as specified in the tag information but not limited to those contexts.
- Where appropriate, simulate misspellings, slang, or typos.
{{- range randomSample .Requirements 5 }}
- {{ . }}
{{- end }}

{{- if .UserInstructions }}
### Additional user instructions:
{{- range .UserInstructions }}
- {{ . }}
{{- end }}
{{- end }}
`

// AnnotatedDataFormat is the JSON‐schema directive to the API.
var AnnotatedDataFormat = map[string]interface{}{
	"type": "json_schema",
	"json_schema": map[string]interface{}{
		"name":        "AnnotatedData",
		"description": "A set of sentences with tokens labeled with specified tags for token classification tasks.",
		"schema": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"sentences": map[string]interface{}{
					"type":        "array",
					"items":       map[string]interface{}{"type": "string"},
					"description": "A sentence containing tokens labeled with specified tags.",
				},
			},
			"required": []string{"sentences"},
		},
	},
}

// AnnotatedDataTmpl is the compiled user template.
var AnnotatedDataTmpl = template.Must(template.New("annotatedData").
	Funcs(template.FuncMap{
		"randomSample": RandomSample,
		"join":         strings.Join,
	}).
	Parse(AnnotatedDataUser))
