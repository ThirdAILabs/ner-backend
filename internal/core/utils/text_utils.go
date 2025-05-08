package utils

import (
	"regexp"
)

var charRegex = regexp.MustCompile(`\S+`)

const DefaultSentenceLength = 100

func SplitTextCustomLength(text string, length int) (sentences []string, startOffsets []int) {
	// we find all the non-whitespace character spans
	// each such span is a token for the NER model
	// we cannot naively split on whitespaces because we need to preserve the token offsets
	idxs := charRegex.FindAllStringIndex(text, -1)

	// every 100 tokens, we create a new sentence
	for tokenIndex := 0; tokenIndex < len(idxs); tokenIndex += length {
		end := min(tokenIndex+length, len(idxs))
		startOffset := idxs[tokenIndex][0]
		endOffset := idxs[end-1][1]
		sentences = append(sentences, text[startOffset:endOffset])
		startOffsets = append(startOffsets, startOffset)
	}
	return
}

func SplitText(text string) (sentences []string, startOffsets []int) {
	return SplitTextCustomLength(text, DefaultSentenceLength)
}
