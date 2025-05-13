package core

import (
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
)

const defaultPresidioThreshold = 0.5

var statelessModelTypes = map[string]struct{}{
	"presidio": {},
}

type Model interface {
	Predict(text string) ([]types.Entity, error)

	Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error

	Save(path string) error

	Release()
}

type ModelLoader func(string) (Model, error)

func IsStatelessModel(modelType string) bool {
	_, exists := statelessModelTypes[modelType]
	return exists
}
