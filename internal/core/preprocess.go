package core

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

const escapedPunct = "\\,\\.\\!\\?;:\\-_" + `"'` + "`" + "\\)\\]\\}\\(\\[\\{"

var (
	patternBeforeSpace = regexp.MustCompile(
		fmt.Sprintf("(?<=\\S)[%s](?=\\s+)", escapedPunct),
	)

	patternAfterSpace = regexp.MustCompile(
		fmt.Sprintf("(\\s+)[%s](?=\\S)", escapedPunct),
	)

	tokenRe = regexp.MustCompile(`\S+(?:\s+|$)`)
)

func replacePunctFollowedBySpace(text string) string {
	result := patternBeforeSpace.ReplaceAllStringFunc(text, func(s string) string {
		return " "
	})
	if utf8.RuneCountInString(result) != utf8.RuneCountInString(text) {
		panic("Length changed in replacePunctFollowedBySpace")
	}
	return result
}

func replacePunctAfterSpace(text string) string {
	result := patternAfterSpace.ReplaceAllStringFunc(text, func(s string) string {
		m := patternAfterSpace.FindStringSubmatch(s)
		return m[1] + " "
	})
	if utf8.RuneCountInString(result) != utf8.RuneCountInString(text) {
		panic("Length changed in replacePunctAfterSpace")
	}
	return result
}

func CleanTextWithSpans(text string) (string, [][2]int) {
	t := replacePunctFollowedBySpace(text)
	t = replacePunctAfterSpace(t)

	spans := make([][2]int, 0)
	tokens := make([]string, 0)

	matches := tokenRe.FindAllStringIndex(t, -1)
	for _, match := range matches {
		start := match[0]
		end := match[1]
		chunk := t[start:end]
		tok := strings.TrimSpace(chunk)
		if tok == "" {
			continue
		}

		spans = append(spans, [2]int{start, start + len(tok)})
		tokens = append(tokens, tok)
	}
	cleaned := strings.Join(tokens, " ")
	return cleaned, spans
}
