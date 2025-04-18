package core

import (
	"fmt"
)

type Entity struct {
	Label string
	Text  string
	Start int // TODO how should this be represented?
	End   int
}

type Model interface {
	Predict(text string) ([]Entity, error)
}

func LoadModel(modelType, path string) (Model, error) {
	switch modelType {
	default:
		return nil, fmt.Errorf("unsupported model type: %s", modelType)
	}
}
