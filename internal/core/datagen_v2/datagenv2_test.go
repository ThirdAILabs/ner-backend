package datagenv2

import (
	"testing"

	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"

	"github.com/stretchr/testify/require"
)

func TestGenerate(t *testing.T) {
	outDir := t.TempDir()
	factory, err := NewDataFactory(outDir)
	require.NoError(t, err)

	tags := []types.TagInfo{
		{
			Name:     "PHONE_NUMBER",
			Desc:     "a phone number",
			Examples: []string{"+1-800-555-1234", "(123) 456-7890"},
		},
		{
			Name: "NAME",
			Desc: "A person's name",
			Examples: []string{
				"John Doe",
				"Jane Smith",
				"Karun Nayar",
			},
		},
		{
			Name:     "ADDRESS",
			Desc:     "A physical address",
			Examples: []string{"123 Main St, Springfield", "456 Elm St, Metropolis"},
		},
	}
	samples := []api.Sample{
		{
			Tokens: []string{"My", "phone", "number", "is", "+1-800-555-1234"},
			Labels: []string{"O", "O", "O", "O", "PHONE_NUMBER"},
		},
		{
			Tokens: []string{"Karun", "Nayar", "lives", "at", "123", "Main", "St,", "Springfield"},
			Labels: []string{"NAME", "NAME", "O", "O", "ADDRESS", "ADDRESS", "ADDRESS", "ADDRESS"},
		},
		{
			Tokens: []string{"Jane", "Smith", "works", "at", "TechCorp"},
			Labels: []string{"NAME", "NAME", "O", "O", "O"},
		},
	}

	opts := GenerateOptions{
		TagsInfo:          tags,
		Samples:           samples,
		RecordsToGenerate: 20,
		RecordsPerLlmCall: 5,
		TestSplit:         0.2,
	}

	train, test, err := factory.Generate(opts)
	require.NoError(t, err)

	// there should be at least 14 train samples and and 2 test samples
	require.GreaterOrEqual(t, len(train), 14)
	require.GreaterOrEqual(t, len(test), 2)
}
