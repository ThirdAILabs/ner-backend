package core

import (
	"fmt"
	"regexp"
	"strings"
)

var (
	punctChars = []string{
		",", ".", "!", "?", ";", ":", "-", "_", "\"", "'", "`",
		")", "]", "}", "(", "[", "{",
	}

	tokenRE = regexp.MustCompile(`\S+(?:\s+|$)`)
)

func replacePunctFollowedBySpace(text string) string {
	originalLen := len(text)

	// Simulate (?<=\S)[punct](?=\s+) - replace punct with space when:
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		char := string(runes[i])

		// Check if current char is punctuation
		if isPunctChar(char) {
			// Check if preceded by non-whitespace (lookbehind simulation)
			hasPrecedingNonSpace := i > 0 && !isSpace(runes[i-1])

			// Check if followed by one or more spaces (lookahead simulation)
			hasFollowingSpaces := false
			if i < len(runes)-1 {
				j := i + 1
				for j < len(runes) && isSpace(runes[j]) {
					j++
				}
				hasFollowingSpaces = j > i+1 // at least one space found
			}

			if hasPrecedingNonSpace && hasFollowingSpaces {
				runes[i] = ' '
			}
		}
	}

	result := string(runes)
	newLen := len(result)
	if originalLen != newLen {
		panic(fmt.Sprintf("Original length: %d, New length: %d", originalLen, newLen))
	}

	return result
}

func replacePunctAfterSpace(text string) string {
	originalLen := len(text)

	// Simulate (\s+)[punct](?=\S) - replace punct with space when:
	runes := []rune(text)
	for i := 0; i < len(runes); i++ {
		char := string(runes[i])

		// Check if current char is punctuation
		if isPunctChar(char) {
			// Check if preceded by one or more spaces
			hasPrecedingSpaces := false
			if i > 0 {
				j := i - 1
				for j >= 0 && isSpace(runes[j]) {
					j--
				}
				hasPrecedingSpaces = j < i-1 // at least one space found
			}

			// Check if followed by non-whitespace (lookahead simulation)
			hasFollowingNonSpace := i < len(runes)-1 && !isSpace(runes[i+1])

			if hasPrecedingSpaces && hasFollowingNonSpace {
				runes[i] = ' '
			}
		}
	}

	result := string(runes)
	newLen := len(result)
	if originalLen != newLen {
		panic(fmt.Sprintf("Original length: %d, New length: %d", originalLen, newLen))
	}

	return result
}

func isPunctChar(char string) bool {
	for _, p := range punctChars {
		if char == p {
			return true
		}
	}
	return false
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f' || r == '\v'
}

func CleanTextWithSpans(text string) (string, [][2]int) {
	t := replacePunctFollowedBySpace(text)
	t = replacePunctAfterSpace(t)

	// Token+whitespace matches so we capture every run of spaces
	spans := make([][2]int, 0)
	var tokens []string

	matches := tokenRE.FindAllStringIndex(t, -1)
	for _, match := range matches {
		start, end := match[0], match[1]
		chunk := t[start:end]
		tok := strings.TrimSpace(chunk)

		if tok == "" {
			continue
		}

		// Calculate exact token boundaries
		tokStart := start
		tokEnd := start + len(tok)

		spans = append(spans, [2]int{tokStart, tokEnd})
		tokens = append(tokens, tok)
	}

	cleaned := strings.Join(tokens, " ")
	return cleaned, spans
}
