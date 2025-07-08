package prompts

import (
	"fmt"
	"math/rand"

	"github.com/openai/openai-go"
)

func RandomSample(src []string, n int) []string {
	length := len(src)
	if n >= length {
		out := make([]string, length)
		copy(out, src)
		return out
	}

	idx := rand.Perm(length)[:n]
	out := make([]string, n)
	for i, j := range idx {
		out[i] = src[j]
	}
	return out
}

func ToResponseFormatUnion(
	format map[string]interface{},
) (openai.ChatCompletionNewParamsResponseFormatUnion, error) {
	jsRaw, ok := format["json_schema"].(map[string]interface{})
	if !ok {
		return openai.ChatCompletionNewParamsResponseFormatUnion{},
			fmt.Errorf("response format missing json_schema key")
	}

	nameVal, ok := jsRaw["name"].(string)
	if !ok {
		return openai.ChatCompletionNewParamsResponseFormatUnion{},
			fmt.Errorf("json_schema.name is not a string")
	}

	descVal, ok := jsRaw["description"].(string)
	if !ok {
		return openai.ChatCompletionNewParamsResponseFormatUnion{},
			fmt.Errorf("json_schema.description is not a string")
	}

	schemaVal, ok := jsRaw["schema"].(map[string]interface{})
	if !ok {
		return openai.ChatCompletionNewParamsResponseFormatUnion{},
			fmt.Errorf("json_schema.schema is not an object")
	}

	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
			JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        nameVal,
				Description: openai.String(descVal),
				Schema:      schemaVal,
			},
		},
	}, nil
}
