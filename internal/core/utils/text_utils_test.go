package utils_test

import (
	"reflect"
	"testing"

	"ner-backend/internal/core/utils"
)

func TestSplitText(t *testing.T) {
	tests := []struct {
		name             string
		text             string
		sentenceLength   int
		wantSentences    []string
		wantStartOffsets []int
	}{
		{
			name:           "Split with custom length 2",
			text:           "hello \n\n world \t\t how are you",
			sentenceLength: 2,
			wantSentences: []string{
				"hello \n\n world",
				"how are",
				"you",
			},
			wantStartOffsets: []int{0, 18, 26},
		},
		{
			name:           "Split with default length",
			text:           "hello .!/////!!??world \n\n\n\n \t\t\t\t\t\t how are you",
			sentenceLength: utils.DefaultSentenceLength,
			wantSentences: []string{
				"hello .!/////!!??world \n\n\n\n \t\t\t\t\t\t how are you",
			},
			wantStartOffsets: []int{0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotSentences, gotStartOffsets := utils.SplitTextCustomLength(tt.text, tt.sentenceLength)
			if !reflect.DeepEqual(gotSentences, tt.wantSentences) {
				t.Errorf("SplitTextWithLength() sentences = %v, want %v", gotSentences, tt.wantSentences)
			}
			if !reflect.DeepEqual(gotStartOffsets, tt.wantStartOffsets) {
				t.Errorf("SplitTextWithLength() startOffsets = %v, want %v", gotStartOffsets, tt.wantStartOffsets)
			}
		})
	}
}
