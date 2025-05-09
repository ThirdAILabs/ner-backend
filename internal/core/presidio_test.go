package core

import (
	"fmt"
	"testing"
)

var model *PresidioModel = nil

func init() {
	if model != nil {
		return
	}
	var err error
	model, err = NewPresidioModel()
	if err != nil {
		panic(fmt.Sprintf("failed to create Presidio model: %v", err))
	}
}

func TestRecognize(t *testing.T) {
	text := "The story of Leo Morgan fitness journey began in Boston witnessed by their national ID 789-67-4567 and visa permit 56482937 Their tranquil abode at 0899 Mark Centers Anthonyfurt, NE 61628 masked the suspense of their mother's maiden name, Russell Drop an email to sonia41@example.net for more."
	entities, err := model.Predict(text)
	if err != nil {
		t.Fatalf("failed to predict: %v", err)
	}

	entitiesDetected
}
