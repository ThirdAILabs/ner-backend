package core

import (
	"fmt"
	"log"
	"ner-backend/internal/core/bolt"
	"ner-backend/internal/core/python"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
	"path/filepath"
)

type ModelType string

const (
	BoltUdt           ModelType = "bolt_udt"
	PythonTransformer ModelType = "python_transformer"
	PythonCnn         ModelType = "python_cnn"
	Presidio          ModelType = "presidio"
	OnnxCnn           ModelType = "onnx_cnn"
	Regex             ModelType = "regex" // Model type for tests
)

func ParseModelType(s string) ModelType {
	m := ModelType(s)
	switch m {
	case BoltUdt, PythonTransformer, PythonCnn, Presidio, OnnxCnn, Regex:
		return m
	default:
		log.Fatalf("invalid model type: %s", s)
		return "" // unreachable, just for compiler
	}
}

const defaultPresidioThreshold = 0.5

var statelessModelTypes = map[ModelType]struct{}{
	Presidio: {},
}

type Model interface {
	Predict(text string) ([]types.Entity, error)

	FinetuneAndSave(taskPrompt string, tags []api.TagInfo, samples []api.Sample, savePath string) error

	Release()
}

type ModelLoader func(string) (Model, error)

func IsStatelessModel(modelType ModelType) bool {
	_, exists := statelessModelTypes[modelType]
	return exists
}

func NewModelLoaders(pythonExec, pluginScript string) map[ModelType]ModelLoader {
	return map[ModelType]ModelLoader{
		BoltUdt: func(modelDir string) (Model, error) {
			return bolt.LoadNER(filepath.Join(modelDir, "model.bin"))
		},
		PythonTransformer: func(modelDir string) (Model, error) {
			cfgJSON := fmt.Sprintf(`{"model_path":"%s","threshold":0.5}`, modelDir)
			return python.LoadPythonModel(
				pythonExec,
				pluginScript,
				"python_combined_ner_model",
				cfgJSON,
			)
		},
		PythonCnn: func(modelDir string) (Model, error) {
			return python.LoadCnnModel(pythonExec, pluginScript, modelDir)
		},
		Presidio: func(_ string) (Model, error) {
			return NewPresidioModel()
		},
		OnnxCnn: func(modelDir string) (Model, error) {
			return LoadOnnxModel(modelDir, pythonExec, pluginScript)
		},
	}
}
