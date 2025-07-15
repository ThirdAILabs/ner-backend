//go:build datagenv2
// +build datagenv2

// only run when tests are run with go test -tags=datagenv2
// this is to avoid having too many llm calls for tests

package datagenv2

import (
	"testing"

	"ner-backend/pkg/api"

	"github.com/stretchr/testify/require"
)

func TestGenerate(t *testing.T) {
	outDir := t.TempDir()
	factory, err := NewDataFactory(outDir)
	require.NoError(t, err)

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

	opts := GenerateOptions{
		TagsInfo:           tags,
		Samples:            samples,
		RecordsToGenerate:  2,
		RecordsPerTemplate: 2,
		TestSplit:          0.5,
	}

	train, test, err := factory.Generate(opts)
	require.NoError(t, err)

	require.Len(t, train, 1)
	require.Len(t, test, 1)
}
