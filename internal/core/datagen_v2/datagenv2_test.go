package datagenv2

import (
	"slices"
	"testing"

	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"

	"github.com/stretchr/testify/require"
)

func TestGenerate(t *testing.T) {
	outDir := t.TempDir()
	factory, err := NewDataFactory(outDir)
	require.NoError(t, err)

	// Picking NAME, ADDRESS, and PHONE_NUMBER from common_tag for testing so that LLM won't have to enhance these tags, saving LLM token cost.
	tags := make([]types.TagInfo, 3)
	for _, commonTag := range types.CommonModelTags {
		if slices.Contains([]string{"NAME", "ADDRESS", "PHONE_NUMBER"}, commonTag.Name) {
			tags = append(tags, commonTag)
		}
		if len(tags) == 3 {
			break
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
}
