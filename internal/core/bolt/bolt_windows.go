//go:build windows
// +build windows

package bolt

import (
	"errors"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
)

type NER int

var (
	ErrBoltNotSupported = errors.New("bolt is not supported on Windows")
)

func LoadNER(path string) (*NER, error) {
	return nil, ErrBoltNotSupported
}

func (ner *NER) Predict(text string) ([]types.Entity, error) {
	return nil, ErrBoltNotSupported
}

func (ner *NER) train(filename string, learningRate float32, epochs int) error {
	return ErrBoltNotSupported

}

func (ner *NER) FinetuneAndSave(taskPrompt string, tags []api.TagInfo, samples []api.Sample, savePath string) error {
	return ErrBoltNotSupported

}

func (ner *NER) Save(path string) error {
	return ErrBoltNotSupported
}

func (ner *NER) Release() {
}

func WindowsFunc() {

}
