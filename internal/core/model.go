package core

import (
	"fmt"
	"os"

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
	"bolt": func(path string) (Model, error) {
		return bolt.LoadNER(path)
	},

	// TODO: replace env vars with a passed-in config
	"python_combined_ner_model": func(path string) (Model, error) {
		configJSON := fmt.Sprintf("{\"model_path\":\"%s\",\"threshold\": 0.5}", path)
		return python.LoadPythonModel(
			os.Getenv("PYTHON_EXECUTABLE_PATH"),
			os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH"),
			"python_combined_ner_model",
			configJSON,
		)
	},

	"presidio": func(arg string) (Model, error) {
		// we ignore `path` (no checkpoint needed) and always use the default threshold
		return &presidioModel{threshold: defaultPresidioThreshold}, nil
	},

	"python_ensemble_ner_model": func(path string) (Model, error) {
		configJSON := fmt.Sprintf("{\"cnn\":{\"model_path\":\"%s/models/ensemble/best_cnn.pth\",\"tokenizer_name\":\"Qwen/Qwen2.5-0.5B\",\"embedding_path\":\"%s/models/ensemble/cnnwrapped_model.pt\"},\"embedding_bag\":{\"checkpoint_path\":\"%s/models/ensemble/bolttorch.pth\"},\"udt\":{\"model_path\":\"%s/models/ensemble/udt_complete.model\"}}", path, path, path, path)
		return python.LoadPythonModel(
			os.Getenv("PYTHON_EXECUTABLE_PATH"),
			os.Getenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH"),
			"python_ensemble_ner_model",
			configJSON,
		)
	},
}

func RegisterModelLoader(modelType string, loader modelLoader) {
	modelLoaders[modelType] = loader
}

func LoadModel(modelType, path string) (Model, error) {
	loader, ok := modelLoaders[modelType]
	if !ok {
		return nil, fmt.Errorf("unknown model type %s", modelType)
	}

	return loader(path)
}

func IsStatelessModel(modelType string) bool {
	_, exists := statelessModelTypes[modelType]
	return exists
}
