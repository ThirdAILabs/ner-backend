package datagenv2

import (
	"testing"

	"ner-backend/pkg/api"

	"github.com/stretchr/testify/require"
)

func TestGenerate(t *testing.T) {
	// Create a temporary output directory
	outDir := t.TempDir()
	factory, err := NewDataFactory(outDir)
	require.NoError(t, err)

	// Define the initial tag and sample
	tags := []TagInfo{
		{
			Name:     "phone_number",
			Desc:     "a phone number",
			Examples: []string{"+1-800-555-1234", "(123) 456-7890"},
		},
	}
	samples := []api.Sample{
		{
			Tokens: []string{"My", "phone", "number", "is", "+1-800-555-1234"},
			Labels: []string{"O", "O", "O", "O", "phone_number"},
		},
	}

	// Configure generation options: generate 2 records, annotate 2 per template, split 50/50
	opts := GenerateOptions{
		TagsInfo:           tags,
		Samples:            samples,
		RecordsToGenerate:  2,
		RecordsPerTemplate: 2,
		TestSplit:          0.5,
	}

	// Execute generation
	train, test, err := factory.Generate(opts)
	require.NoError(t, err)

	// Expect one sample in train and one in test
	require.Len(t, train, 1)
	require.Len(t, test, 1)
}
