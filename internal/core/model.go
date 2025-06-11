package core

import (
	"fmt"
	"ner-backend/internal/core/bolt"
	"ner-backend/internal/core/python"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
	"path/filepath"
)

// ModelType represents the type of NER model
type ModelType string

// Available model types
const (
	BoltUdt           ModelType = "bolt_udt"
	PythonTransformer ModelType = "python_transformer"
	PythonCnn         ModelType = "python_cnn"
	Presidio          ModelType = "presidio"
	OnnxCnn           ModelType = "onnx_cnn"
)

const defaultPresidioThreshold = 0.5

var statelessModelTypes = map[ModelType]struct{}{
	Presidio: {},
}

type Model interface {
	Predict(text string) ([]types.Entity, error)

	Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error

	Save(path string) error

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
			cfgJSON := fmt.Sprintf(`{"model_path":"%s/cnn_model.pth", "tokenizer_path":"%s/qwen_tokenizer"}`, modelDir, modelDir)
			return python.LoadPythonModel(
				pythonExec,
				pluginScript,
				"python_cnn_ner_model",
				cfgJSON,
			)
		},
		Presidio: func(_ string) (Model, error) {
			return NewPresidioModel()
		},
		OnnxCnn: func(modelDir string) (Model, error) {
			return LoadOnnxModel(modelDir)
		},
	}
}
