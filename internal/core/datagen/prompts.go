package datagen

import (
	"text/template"
)

// ----------------- Prompt for generating tag values -------------------
type tagValuePromptFields struct {
	NumValues int
	Tag       string
}

const tagValuePrompt = `You are an expert in generating diverse and realistic data samples. 

Please generate {{ .NumValues }} unique and varied values for the "{{ .Tag }}" named entity. Ensure the values reflect realistic examples in the context of a US citizen.

Requirements:
- Each value must start on a new line.
- **DO NOT** add any headers, footers, bullet points, or any other formatting.
- **DO NOT** use bullet points, quotes, emojis, or any prefix.
- Include a wide range of formats and variations typical for the "{{ .Tag }}" entity.

Make the values as diverse and representative as possible.`

var tagValuePromptTmpl = template.Must(template.New("tagValuePrompt").Parse(tagValuePrompt))

// --------------------------------------------------------------------

// ------------------ prompt to generate templates from user samples ------------------------
type templateFromSamplePromptFields struct {
	NumTemplates int
	TagsName     []string
	Samples      []string
}

// Note: the original prompt included random prompts in the "Key Requirements" section, however this is removed because
// they don't seem relevant to NER
const templateFromSamplePrompt = `The goal is to create a dataset for entity recognition.

Please generate {{ .NumTemplates }} diverse templates using the following tags:
{{ range .TagsName }}- [{{ . }}]
{{ end }}

Here are some example templates using these tags:
{{ range .Samples }}
{{ . }}
{{ end }}

Key Requirements:
- The generated templates should follow similar grammatical or semantic patterns as the samples but must not be identical.
- Include words that resemble or suggest the tag contextually but are not actual tagged entities (to introduce ambiguity).
- Try to include multiple tags in each template when it makes sense.

Output format:
- Each template should appear on a new line.
- **DO NOT** add any headers, footers, bullet points, or any other formatting.
- **DO NOT** use bullet points, quotes, emojis, or any prefix before each template.

IMPORTANT:
- These templates will later be populated with actual tag values.
- Ensure each template remains logical and coherent after the tags are replaced.`

var templateFromSamplePromptTmpl = template.Must(template.New("templateFromSamplePrompt").Parse(templateFromSamplePrompt))

// --------------------------------------------------------------------
