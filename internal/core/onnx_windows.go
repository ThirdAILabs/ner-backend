//go:build windows

package core

import (
	"errors"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
)

var ErrOnnxNotSupportedOnWindows = errors.New("ONNX models are not supported on Windows")

type OnnxModel struct{}

func LoadOnnxModel(modelDir string) (Model, error) {
	return nil, ErrOnnxNotSupportedOnWindows
}

func (m *OnnxModel) Predict(text string) ([]types.Entity, error) {
	return nil, ErrOnnxNotSupportedOnWindows
}

func (m *OnnxModel) FinetuneAndSave(taskPrompt string, tags []api.TagInfo, samples []api.Sample, savePath string) error {
	return ErrOnnxNotSupportedOnWindows
}

func (m *OnnxModel) Release() {
	// no-op
}
