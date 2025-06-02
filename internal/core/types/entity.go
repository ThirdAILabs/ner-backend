package types

import (
	"fmt"
	"regexp"
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

func extractBetweenText(e1, e2 Entity) (string, error) {
	if e2.Start-e1.End > len(e1.RContext)+len(e2.LContext) {
		return "", fmt.Errorf("gap between entities is too large to reconstruct")
	}
	gapSize := e2.Start - e1.End

	if gapSize <= len(e1.RContext) {
		// Gap fits entirely within e1's RContext
		return e1.RContext[:gapSize], nil
	} else {
		// Need to use both RContext and LContext
		betweenText := e1.RContext

		remainingGap := gapSize - len(e1.RContext)
		if remainingGap <= len(e2.LContext) {
			startIdx := len(e2.LContext) - remainingGap
			betweenText += e2.LContext[startIdx:]
		} else {
			// Gap is too large, can't reconstruct
			return "", fmt.Errorf("gap between entities is too large to reconstruct")
		}
		return betweenText, nil
	}
}

func AreEntitiesAdjacent(e1, e2 Entity) (bool, string) {
	if e2.Start-e1.End > len(e1.RContext)+len(e2.LContext) {
		return false, ""
	}
	textBetween, _ := extractBetweenText(e1, e2)

	punctuationAndWhitespaceRegex := regexp.MustCompile(`^[!\x22#$%&'()*+,\-./:;<=>?@\[\\\]^_\x60{|}~\s]*$`)
	return punctuationAndWhitespaceRegex.MatchString(textBetween), textBetween
}
