package core

import (
	"fmt"
	"ner-backend/internal/core/bolt"
	"ner-backend/internal/core/python"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
	"os"
)

type Model interface {
	Predict(text string) ([]types.Entity, error)

	Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error

	Save(path string) error

	Release()
}

type modelLoader func(string) (Model, error)

var modelLoaders map[string]modelLoader = map[string]modelLoader{
	"bolt": func(path string) (Model, error) { return bolt.LoadNER(path) },
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

type presidioModel struct {
	threshold float64
}

func (m *presidioModel) Predict(text string) ([]types.Entity, error) {
	results := analyze(text, m.threshold)
	out := make([]types.Entity, 0, len(results))
	for _, r := range results {
		out = append(out, types.Entity{
			Text:  r.Match,
			Label: r.EntityType,
			Start: r.Start,
			End:   r.End,
		})
	}
	return out, nil
}

func (m *presidioModel) Release() {}
