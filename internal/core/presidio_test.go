package core

import (
	"fmt"
	"ner-backend/internal/core/types"
	"testing"

	"github.com/stretchr/testify/assert"
)

var model *PresidioModel = nil

func init() {
	var err error
	model, err = NewPresidioModel()
	if err != nil {
		panic(fmt.Sprintf("fail	ed to create Presidio model: %v", err))
	}
}

func TestRecognize(t *testing.T) {
	text := "The story of Leo Morgan fitness journey began in Boston witnessed by their national ID 789-67-4567 and visa permit 56482937 Their tranquil abode at 0899 Mark Centers Anthonyfurt, NE 61628 masked the suspense of their mother's maiden name, Russell Drop an email to sonia41@example.net for more."
	entities, err := model.Predict(text)
	if err != nil {
		t.Fatalf("failed to predict: %v", err)
	}

	assert.ElementsMatch(t, entities, []types.Entity{
		{
			Label:    "SSN",
			Text:     "789-67-4567",
			Start:    87,
			End:      98,
			LContext: "y their national ID ",
			RContext: " and visa permit 564",
		},
		{
			Label:    "EmailRecognizer",
			Text:     "sonia41@example.net",
			Start:    264,
			End:      283,
			LContext: "ll Drop an email to ",
			RContext: " for more.",
		},
		{
			Label:    "UrlRecognizer",
			Text:     "example.net",
			Start:    272,
			End:      283,
			LContext: "an email to sonia41@",
			RContext: " for more.",
		},
	}, "Incorrect entities recognized by the presidio model")
}
