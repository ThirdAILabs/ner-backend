package types

import (
	"log/slog"
	"strings"
)

const contextLength = 20

type Entity struct {
	Label    string
	Text     string
	Start    int
	End      int
	LContext string
	RContext string
}

func CreateEntity(label, context string, start, end int) Entity {
	runes := []rune(context)

	if start < 0 {
		start = 0
	}
	if end > len(runes) {
		end = len(runes)
	}

	text := string(runes[start:end])

	leftStart := start - contextLength
	if leftStart < 0 {
		leftStart = 0
	}
	lctx := string(runes[leftStart:start])

	rightEnd := end + contextLength
	if rightEnd > len(runes) {
		rightEnd = len(runes)
	}
	rctx := string(runes[end:rightEnd])

	entity := Entity{
		Label:    label,
		Text:     strings.ToValidUTF8(text, ""),
		Start:    start,
		End:      end,
		LContext: strings.ToValidUTF8(lctx, ""),
		RContext: strings.ToValidUTF8(rctx, ""),
	}
	slog.Info("CreateEntity", "entity", entity)
	return entity
}
