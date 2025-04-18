package core

import (
	"fmt"
	"ner-backend/internal/core/bolt"
	"ner-backend/internal/core/types"
	// "ner-backend/internal/core/bolt"
)

type Model interface {
	Predict(text string) ([]types.Entity, error)

	Release()
}

func LoadModel(modelType, path string) (Model, error) {
	switch modelType {
	case "bolt":
		return bolt.LoadNER(path)
	default:
		return nil, fmt.Errorf("unsupported model type: %s", modelType)
	}
}
