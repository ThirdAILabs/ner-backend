package types

import "strings"

const contextLength = 20

type Entity struct {
	Label    string
	Text     string
	Start    int
	End      int
	LContext string
	RContext string
}

func (e *Entity) UpdateContext(text string) {
	if e.LContext == "" {
		e.LContext = strings.ToValidUTF8(text[max(0, e.Start-contextLength):e.Start], "")
	}
	if e.RContext == "" {
		e.RContext = strings.ToValidUTF8(text[e.End-1:min(len(text), e.End+contextLength)], "")
	}
}
