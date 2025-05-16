package utils

import (
	"regexp"
)

var charRegex = regexp.MustCompile(`\S+`)

const DefaultSentenceLength = 100

// byteOffsetToRuneIndex turns a byte‐offset into the corresponding rune‐index.
func byteOffsetToRuneIndex(s string, byteOff int) int {
	if byteOff <= 0 {
		return 0
	}
	runeIdx := 0
	for i := range s {
		if i >= byteOff {
			break
		}
		runeIdx++
	}
	return runeIdx
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// SplitTextCustomLength returns sentences *and* their START OFFSETS
// in terms of runes, so we can feed them (and those offsets) directly
// into CreateEntity (which slices by rune).
func SplitTextCustomLength(text string, length int) (sentences []string, startOffsets []int) {
	// convert once
	runes := []rune(text)

	// find token spans in byte‐space
	idxs := charRegex.FindAllStringIndex(text, -1)

	for tokenIndex := 0; tokenIndex < len(idxs); tokenIndex += length {
		end := min(tokenIndex+length, len(idxs))

		// byte offsets for this chunk
		bStart := idxs[tokenIndex][0]
		bEnd := idxs[end-1][1]

		// convert them into rune offsets
		rStart := byteOffsetToRuneIndex(text, bStart)
		rEnd := byteOffsetToRuneIndex(text, bEnd)

		// slice the rune‐array, back to string
		sentences = append(sentences, string(runes[rStart:rEnd]))
		startOffsets = append(startOffsets, rStart)
	}

	return
}

func SplitText(text string) (sentences []string, startOffsets []int) {
	return SplitTextCustomLength(text, DefaultSentenceLength)
}
