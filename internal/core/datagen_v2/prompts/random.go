package prompts

import (
	"fmt"
	"math/rand"

	"github.com/openai/openai-go"
)

// RandomSample returns up to n random elements from src.
// If len(src) ≤ n, it simply returns a copy of src.
func RandomSample(src []string, n int) []string {
	length := len(src)
	if n >= length {
		// return a copy so template funcs can’t mutate your original
		out := make([]string, length)
		copy(out, src)
		return out
	}

	// create a permutation of [0..length)
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
	// Look up the inner "json_schema" object
	jsRaw, ok := format["json_schema"].(map[string]interface{})
	if !ok {
		return openai.ChatCompletionNewParamsResponseFormatUnion{},
			fmt.Errorf("response format missing json_schema key")
	}

	// Extract name
	nameVal, ok := jsRaw["name"].(string)
	if !ok {
		return openai.ChatCompletionNewParamsResponseFormatUnion{},
			fmt.Errorf("json_schema.name is not a string")
	}

	// Extract description
	descVal, ok := jsRaw["description"].(string)
	if !ok {
		return openai.ChatCompletionNewParamsResponseFormatUnion{},
			fmt.Errorf("json_schema.description is not a string")
	}

	// Extract the actual schema map
	schemaVal, ok := jsRaw["schema"].(map[string]interface{})
	if !ok {
		return openai.ChatCompletionNewParamsResponseFormatUnion{},
			fmt.Errorf("json_schema.schema is not an object")
	}

	// Build and return the union
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
