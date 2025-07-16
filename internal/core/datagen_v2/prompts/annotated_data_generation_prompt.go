package prompts

import (
	"html/template"
	"strings"
)

const AnnotatedDataSystem = `# You are a data generation assistant specialized in creating high-quality, diverse, and realistic text samples for token classification tasks.
## Goal:
Your goal is to generate natural sentences where multiple types of entities (tags) appear in varied and realistic linguistic contexts. 
`

const AnnotatedDataUser = `## Task: Generate {{ .K }} diverse and realistic sentences containing tokens labeled for the following tags.

### Tags information:
{{- range .TagInfo }}
**Tag name:** {{ .Name }}
**Tag description:** {{ .Desc }}
{{- if gt (len .Examples) 0 }}
**Tag examples:** {{ join (randomSample .Examples 10) ", " }}
{{- end }}
{{- if gt (len .Contexts) 0 }}
**Tag context:** {{ join (randomSample .Contexts 4) ", " }}
{{- end }}
{{ end }}
> Token-Tag pair should be enclosed in this tagging format : ##tokens##TAG##
{{ if .Feedback -}}
# Below are some Contextual examples.
{{- range randomSample .Feedback 4 }}
- {{ . }}
{{- end }}
{{- else }}
# Below are some examples of sentences with tokens and their tags:
- "##Karun naiyar##NAME## lives at ##123 Main St, Springfield##ADDRESS##, and his birthday is ##January 1, 1990##DATE##."
- "Working at an unknown company, located in ##Los Angeles##ADDRESS##, and ##Maria Gomez##NAME## was born on ##March 15, 1985##DATE##."
{{- end }}

## Requirements:
{{- range .Requirements }}
- {{ . }}
{{- end }}

{{- if .UserInstructions }}
### Additional user instructions:
{{- range .UserInstructions }}
- {{ . }}
{{- end }}
{{- end }}
`

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

var AnnotatedDataTmpl = template.Must(template.New("annotatedData").
	Funcs(template.FuncMap{
		"randomSample": RandomSample,
		"join":         strings.Join,
	}).
	Parse(AnnotatedDataUser))

type AnnotatedData struct {
	Sentences []string `json:"sentences"`
}

func (a *AnnotatedData) Clean() *AnnotatedData {
	cleaned := make([]string, 0, len(a.Sentences))
	for _, s := range a.Sentences {
		if t := strings.TrimSpace(s); t != "" {
			cleaned = append(cleaned, t)
		}
	}
	a.Sentences = cleaned
	return a
}
