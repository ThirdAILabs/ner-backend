package core

import (
	"fmt"
	"ner-backend/internal/core/bolt"
	"ner-backend/internal/core/python"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
	"path/filepath"
)

const defaultPresidioThreshold = 0.5

var statelessModelTypes = map[string]struct{}{
	"presidio": {},
}

type Model interface {
	Predict(text string) ([]types.Entity, error)

	FinetuneAndSave(taskPrompt string, tags []api.TagInfo, samples []api.Sample, savePath string) error

	Release()
}

type ModelLoader func(string) (Model, error)

func IsStatelessModel(modelType string) bool {
	_, exists := statelessModelTypes[modelType]
	return exists
}

func NewModelLoaders(pythonExec, pluginScript string) map[string]ModelLoader {

	return map[string]ModelLoader{
		"bolt": func(modelDir string) (Model, error) {
			return bolt.LoadNER(filepath.Join(modelDir, "model.bin"))
		},
		"transformer": func(modelDir string) (Model, error) {
			cfgJSON := fmt.Sprintf(`{"model_path":"%s","threshold":0.5}`, modelDir)
			return python.LoadPythonModel(
				pythonExec,
				pluginScript,
				"python_combined_ner_model",
				cfgJSON,
			)
		},
		"cnn": func(modelDir string) (Model, error) {
			return python.LoadCnnModel(pythonExec, pluginScript, modelDir)
		},
		"presidio": func(_ string) (Model, error) {
			return NewPresidioModel()
		},
		"onnx": func(modelDir string) (Model, error) {
			return LoadOnnxModel(modelDir, pythonExec, pluginScript)
		},
	}
}
