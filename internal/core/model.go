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

type modelLoader func(string) (Model, error)

var modelLoaders map[string]modelLoader = map[string]modelLoader{
	"bolt": func(path string) (Model, error) { return bolt.LoadNER(path) },
}

func RegisterModelLoader(modelType string, loader modelLoader) {
	if _, exists := modelLoaders[modelType]; exists {
		panic(fmt.Sprintf("model type %s already registered", modelType))
	}
	modelLoaders[modelType] = loader
}

func LoadModel(modelType, path string) (Model, error) {
	loader, ok := modelLoaders[modelType]
	if !ok {
		return nil, fmt.Errorf("unknown model type %s", modelType)
	}

	return loader(path)
}
