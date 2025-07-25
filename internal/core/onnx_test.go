package core

import (
	"testing"

	"github.com/daulet/tokenizers"
	"github.com/stretchr/testify/assert"
)

func TestCRF(t *testing.T) {
	// Test case is generated by randomly generating transistion probs and emissions and running torchcrf decoding.

	crf := CRF{
		Transitions: [][]float32{
			{0.8, 0.5, 0.2, 0.1},
			{0.2, 0.4, 0.1, 0.3},
			{0.3, 0.6, 0.9, 0.3},
			{0.9, 0.9, 0.9, 0.3},
		},
		StartProbs: []float32{0.7, 0.8, 0.0, 0.5},
		EndProbs:   []float32{0.4, 0.1, 0.3, 0.2},
	}

	emissions := [][][]float32{
		{
			{0.2, 0.7, 0.1, 0.9},
			{0.1, 0.6, 0.8, 0.8},
			{0.8, 0.9, 0.3, 0.1},
			{0.9, 0.8, 0.3, 0.9},
		}, {
			{0.6, 0.0, 0.5, 0.7},
			{0.1, 0.4, 0.1, 0.6},
			{0.3, 0.7, 0.3, 0.0},
		}, {
			{0.5, 0.6, 0.7, 0.5},
			{0.9, 0.5, 0.4, 0.3},
			{0.0, 0.8, 0.9, 0.4},
			{0.8, 0.7, 0.5, 0.8},
			{0.7, 0.5, 0.8, 0.1},
		},
	}

	expectedSeqs := [][]int{
		{1, 3, 0, 0},
		{3, 3, 1},
		{3, 2, 2, 2, 2},
	}

	for i, emissionsSeq := range emissions {
		seq := crf.ViterbiDecode(emissionsSeq)
		assert.Equal(t, expectedSeqs[i], seq)
	}
}

func TestGetWordIDs(t *testing.T) {
	{
		tokenOffsets := []tokenizers.Offset{{0, 3}, {4, 6}, {7, 10}}
		wordOffsets := [][2]int{{0, 3}, {4, 6}, {7, 10}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{0, 1, 2}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{0, 2}, {3, 5}, {6, 8}}
		wordOffsets := [][2]int{{0, 3}, {4, 6}, {7, 10}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{0, 1, 2}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{0, 2}, {3, 5}, {6, 8}}
		wordOffsets := [][2]int{{0, 3}, {4, 6}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{0, 1, -1}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{0, 2}, {5, 7}}
		wordOffsets := [][2]int{{3, 5}, {8, 10}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{-1, -1}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{3, 5}, {8, 10}}
		wordOffsets := [][2]int{{0, 2}, {5, 7}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{-1, -1}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{0, 3}, {4, 6}, {7, 10}}
		wordOffsets := [][2]int{{0, 2}, {3, 5}, {6, 8}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{0, 1, 2}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{0, 1}, {2, 4}, {7, 10}}
		wordOffsets := [][2]int{{0, 3}, {5, 7}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{0, 0, -1}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{0, 1}, {3, 5}, {8, 10}}
		wordOffsets := [][2]int{{0, 3}, {6, 7}, {7, 10}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{0, -1, 2}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{0, 3}, {4, 6}, {7, 10}, {11, 13}}
		wordOffsets := [][2]int{{0, 3}, {5, 8}, {12, 14}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{0, 1, 1, 2}, result)
	}

	{
		tokenOffsets := []tokenizers.Offset{{0, 3}, {4, 6}, {7, 10}}
		wordOffsets := [][2]int{}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{-1, -1, -1}, result)
	}
	{
		tokenOffsets := []tokenizers.Offset{}
		wordOffsets := [][2]int{{0, 3}, {4, 6}, {7, 10}}
		result := getWordIds(wordOffsets, tokenOffsets)
		assert.Equal(t, []int{}, result)
	}
}
