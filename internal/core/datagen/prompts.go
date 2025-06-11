package datagen

import (
	"ner-backend/pkg/api"
	"text/template"
)

type tagValuePromptFields struct {
	TaskPrompt  string
	NumValues   int
	Tag         string
	Description string
	Examples    []string
}

const tagValuePrompt = `You possess deep expertise in {{ .TaskPrompt }}. Please generate unique {{ .NumValues }} diverse values for the {{ .Tag }} named entity. Below are some values of the {{ .Tag }} entity:
[{{ range $i, $v := .Examples }}{{ if $i }}, {{ end }}{{ $v }}{{ end }}]

Description of the tag {{ .Tag }}:
{{ .Description }}
Also Cover a wide variations in the format of the tag {{ .Tag }} value.
Additionally, aim to cover a wide range of variations within the {{ .Tag }} entity to ensure the data is as varied and representative as possible.

VERY IMPORTANT:
-  Ensure each value starts on a new line without any bullet points, prefixes, quotes, or emojis at the beginning of each sentence.
-  Make sure that the samples are relevant to the context of US citizen.`

var tagValuePromptTmpl = template.Must(template.New("tagValuePrompt").Parse(tagValuePrompt))

type templateFromTagPromptFields struct {
	TaskPrompt   string
	NumTemplates int
	TagsInfo     []api.TagInfo
}

// Note: the original prompt included random prompts in the "Key Requirements" section, however this is removed because
// they don't seem relevant to NER
const templateFromTagPrompt = `The goal is to create a dataset for entity recognition. Please generate {{ .NumTemplates }} templates associated with given below tags for {{ .TaskPrompt }}
            
Tags with their description and example:
{{ range .TagsInfo }}
Tag: {{ .Name }}
Description: {{ .Description }}.
Examples: [{{ range $i, $v := .Examples }}{{ if $i }}, {{ end }}{{ $v }}{{ end }}] not limited to given but variations as well.


{{ end }}

For example, here are some templates for the tags [CARD_NUMBER, CARDHOLDER_NAME, EXPIRATION_DATE] on the domain of payment information.
- [CARDHOLDER_NAME] and his friend john tried to dupe the credit card company by reporting their transaction with the card [CARD_NUMBER] as fradulent on 9th august.
- The card was expired in december but the expiration date mentioned was [EXPIRATION_DATE].

Key Requirements:
- Include words that could be interpreted as tag but are actually not, as depicted in the above examples.
- Try to include many tags in each sentences.

Output format:
-  Each template should be in a newline.
-  DO NOT include any bulleting, header/footer, enumeration, prefix/suffix or any process involved.

** IMPORTANT POINT:
-  These templates would be filled later so make sure these templates would make sense after being filled. Here is one incorrect & correct templates for the tag [MEDICAL_INFO]
      Incorrect template: My [MEDICAL_INFO] should remain confidential to protect my personal interest.
      Correct template: My condition due to [MEDICAL_INFO] should remain confidential to protect my personal interest.`

var templateFromTagPromptTmpl = template.Must(template.New("templateFromTagPrompt").Parse(templateFromTagPrompt))

type templateFromSamplePromptFields struct {
	templateFromTagPromptFields
	Sample string
}

// Note: the original prompt included random prompts in the "Key Requirements" section, however this is removed because
// they don't seem relevant to NER
const templateFromSamplePrompt = `The goal is to create a dataset for entity recognition. Please generate {{ .NumTemplates }} templates associated with given below tags for {{ .TaskPrompt }}
            
Tags with their description and example:
{{ range .TagsInfo }}
Tag: {{ .Name }}
Description: {{ .Description }}.
Examples: [{{ range $i, $v := .Examples }}{{ if $i }}, {{ end }}{{ $v }}{{ end }}] not limited to given but variations as well.


{{ end }}

Here is a sample with the specified tags :
{{ .Sample }}
Key Requirements:
- Include words that could be interpreted as tag but are actually not, as depicted in the above examples.
- Generate samples that are somewhat similar in grammatical or semantical structure but not exactly the same as the example given above.

Output format:
-  Each template should be in a newline.
-  DO NOT include any bulleting, header/footer or enumeration. Do not include any quotes or emojis.
** IMPORTANT POINT:
-  These templates would be filled later so make sure these templates would make sense after being filled. Here is one incorrect & correct templates for the tags [MEDICAL_INFO]
      Incorrect templates: My [MEDICAL_INFO] should remain confidential to protect my personal interest.
      Correct templates: My condition due to [MEDICAL_INFO] should remain confidential to protect my personal interest.`

var templateFromSamplePromptTmpl = template.Must(template.New("templateFromSamplePrompt").Parse(templateFromSamplePrompt))

type extendedDescriptionPromptFields struct {
	Name        string
	Description string
	Examples    []string
}

const extendedDescriptionPrompt = `The goal is to get a comprehensive description of the given attribute. Please generate an extended description of the attribute {{ .Name }}. Below are the user's example and description of the attribute

Attribute with it's user description and example:
Name: {{ .Name }}
User description: {{ .Description }}
Examples: [{{ range $i, $v := .Examples }}{{ if $i }}, {{ end }}{{ $v }}{{ end }}]

For example,
Attribute: PAN
User description: Patterns that match primary account number formats (typically 16-digit numbers)
Examples: [4587-8918-4578-2688, 124556893256]

==> Extended-description: A Primary Account Number (PAN) is a 16-digit number on credit/debit cards, including the Issuer Identification Number (IIN) and account number, following patterns like Visa (4###) and MasterCard (5###).

Output format:
-  Only output the extended description within 50-70 words
-  DO NOT include any bulleting, header/footer, enumeration, prefix/suffix or any process involved. Do not include any quotes or emojis.`

var extendedDescriptionPromptTmpl = template.Must(template.New("extendedDescriptionPrompt").Parse(extendedDescriptionPrompt))
