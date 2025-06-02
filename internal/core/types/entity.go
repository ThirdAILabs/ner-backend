package types

import (
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

func CreateEntity(label string, context string, start int, end int) Entity {
	entity := Entity{
		Label:    label,
		Text:     strings.ToValidUTF8(context[start:end], ""),
		Start:    start,
		End:      end,
		LContext: strings.ToValidUTF8(context[max(0, start-contextLength):start], ""),
		RContext: strings.ToValidUTF8(context[end:min(len(context), end+contextLength)], ""),
	}
	return entity
}
