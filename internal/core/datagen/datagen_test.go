//go:build datagen
// +build datagen

// only run when tests are run with go test -tags=datagen
// this is to avoid having too many llm calls for tests

package datagen

import (
	"ner-backend/pkg/api"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDatagen(t *testing.T) {
	train, test, err := GenerateData(DatagenOpts{
		TaskPrompt: "Generate training data to detect phone numbers",
		Tags:       []api.TagInfo{{Name: "phone_number", Description: "a phone number", Examples: []string{"+1-800-555-1234", "(123) 456-7890"}}},
		Samples: []api.Sample{{
			Tokens: []string{"My", "phone", "number", "is", "+1-800-555-1234"},
			Labels: []string{"O", "O", "O", "O", "phone_number"},
		}},
		NumValuesPerTag:    10,
		SamplesToGenerate:  10,
		SamplesPerTemplate: 2,
		GenerateAtOnce:     2,
		TemplatesPerSample: 3,
		TestSplit:          0.3,
	})
	require.NoError(t, err)

	require.NotEmpty(t, train)
	require.NotEmpty(t, test)
}
