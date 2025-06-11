package core

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	punctChars = "-,.!?:_\"'`)]}([{"

	replacePunctFollowedBySpaceRE = regexp.MustCompile(fmt.Sprintf(`(\S)[%s](\s)`, regexp.QuoteMeta(punctChars)))
	replaceSpaceFollowedByPunctRE = regexp.MustCompile(fmt.Sprintf(`(\s)[%s](\S)`, regexp.QuoteMeta(punctChars)))

	tokenRE = regexp.MustCompile(`\S+`)
)

func replacePunctFollowedBySpace(text string) string {
	newText := replacePunctFollowedBySpaceRE.ReplaceAllString(text, "$1 $2")
	if len(text) != len(newText) {
		panic(fmt.Sprintf("Text: '%s' Original length: %d, New length: %d", text, len(text), len(newText)))
	}

	return newText
}

func replaceSpaceFollowedByPunct(text string) string {
	newText := replaceSpaceFollowedByPunctRE.ReplaceAllString(text, "$1 $2")
	if len(text) != len(newText) {
		panic(fmt.Sprintf("Text: '%s' Original length: %d, New length: %d", text, len(text), len(newText)))
	}

	return newText
}

func CleanTextWithSpans(text string) (string, [][2]int, [][2]int) {
	text = replacePunctFollowedBySpace(text)
	text = replaceSpaceFollowedByPunct(text)

	// Token+whitespace matches so we capture every run of spaces
	originalSpans := make([][2]int, 0)
	cleanedSpans := make([][2]int, 0)
	cleanedText := strings.Builder{}

	matches := tokenRE.FindAllStringIndex(text, -1)
	for _, match := range matches {
		originalStart, originalEnd := match[0], match[1]
		token := text[originalStart:originalEnd]

		cleanedText.WriteByte(' ')
		cleanedStart := cleanedText.Len()
		cleanedText.WriteString(token)
		cleanedEnd := cleanedText.Len()

		originalSpans = append(originalSpans, [2]int{originalStart, originalEnd})
		cleanedSpans = append(cleanedSpans, [2]int{cleanedStart, cleanedEnd})
	}

	return cleanedText.String(), originalSpans, cleanedSpans
}
