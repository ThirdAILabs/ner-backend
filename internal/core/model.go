package core

import (
	"fmt"
	"path/filepath"
	"strings"

	"ner-backend/internal/core/bolt"
	"ner-backend/internal/core/python"
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

type modelLoader func(string) (Model, error)

var modelLoaders = map[string]modelLoader{
	"bolt": func(modelDir string) (Model, error) {
		return bolt.LoadNER(filepath.Join(modelDir, "model.bin"))
	},

	// TODO: replace env vars with a passed-in config
	"transformer": func(path string) (Model, error) {
		configJSON := fmt.Sprintf("{\"model_path\":\"%s\",\"threshold\": 0.5}", path)
		return python.LoadPythonModel(
			"python",
			"plugin/plugin-python/plugin.py",
			"python_combined_ner_model",
			configJSON,
		)
	},

	"presidio": func(arg string) (Model, error) {
		// we ignore `path` (no checkpoint needed) and always use the default threshold
		return NewPresidioModel()
	},

	"cnn": func(path string) (Model, error) {
		configJSON := fmt.Sprintf("{\"model_path\":\"%s/cnn_model.pth\"}", path)
		return python.LoadPythonModel(
			"python",
			"plugin/plugin-python/plugin.py",
			"python_cnn_ner_model",
			configJSON,
		)
	},
}

func RegisterModelLoader(modelType string, loader modelLoader) {
	modelLoaders[modelType] = loader
}

func LoadModel(modelType, modelDir string) (Model, error) {
	loader, ok := modelLoaders[modelType]
	if !ok {
		return nil, fmt.Errorf("unknown model type %s", modelType)
	}

	modelDir = strings.TrimSuffix(modelDir, "/")

	return loader(modelDir)
}

func IsStatelessModel(modelType string) bool {
	_, exists := statelessModelTypes[modelType]
	return exists
}
