package bolt

// #cgo linux LDFLAGS: -L./lib/linux_amd64 -L./lib/linux_arm64 -lthirdai -lrocksdb -lutf8proc -lspdlog -fopenmp
// #cgo darwin LDFLAGS: -L./lib/macos_arm64 -lthirdai -lrocksdb -lutf8proc -lspdlog -L/opt/homebrew/opt/libomp/lib/ -lomp
// #cgo CFLAGS: -O3
// #cgo CXXFLAGS: -O3 -fPIC -std=c++17 -I./include -fvisibility=hidden
// #include "binding.h"
// #include <stdlib.h>
import "C"
import (
	"encoding/csv"
	"errors"
	"io"
	"ner-backend/internal/core/types"
	"ner-backend/internal/core/utils"
	"ner-backend/pkg/api"
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

type NER struct {
	model *C.NER_t
}

func LoadNER(path string) (*NER, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var err *C.char
	model := C.NER_load(cPath, &err)
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		return nil, errors.New(C.GoString(err))
	}
	return &NER{model: model}, nil
}

func (ner *NER) Predict(text string) ([]types.Entity, error) {
	sentences, startOffsets := utils.SplitText(text)

	cSentences := newStringList(sentences)
	defer C.StringList_free(cSentences)

	var err *C.char
	cResults := C.NER_predict(ner.model, cSentences, &err)
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		return nil, errors.New(C.GoString(err))
	}
	defer C.Results_free(cResults)

	batchSize := C.Results_batch_size(cResults)
	results := make([]types.Entity, 0)
	for i := C.uint(0); i < batchSize; i++ {
		itemResultsLen := C.Results_len(cResults, i)
		globalOffset := startOffsets[int(i)]
		for j := C.uint(0); j < itemResultsLen; j++ {
			start := int(C.Results_start(cResults, i, j)) + globalOffset
			end := int(C.Results_end(cResults, i, j)) + globalOffset
			entity := types.CreateEntity(
				C.GoString(C.Results_label(cResults, i, j)),
				text,
				start,
				end,
			)
			results = append(results, entity)
		}
	}

	return results, nil
}

func (ner *NER) train(filename string, learningRate float32, epochs int) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))

	var err *C.char
	C.NER_train(ner.model, cFilename, C.float(learningRate), C.uint(epochs), &err)
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		return errors.New(C.GoString(err))
	}
	return nil
}

func (ner *NER) Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error {
	var cTokensCol, cTagsCol *C.char
	C.NER_source_target_cols(ner.model, &cTokensCol, &cTagsCol)
	tokensCol := C.GoString(cTokensCol)
	tagsCol := C.GoString(cTagsCol)
	C.NER_source_target_cols_free(cTokensCol, cTagsCol)

	tmpFile, err := os.CreateTemp("", "ner_finetune_*.csv")
	if err != nil {
		return err
	}
	defer tmpFile.Close()
	defer os.Remove(tmpFile.Name())

	if err := writeSamplesToCSV(tmpFile, samples, tokensCol, tagsCol); err != nil {
		return err
	}

	// call train with default hyperparameters
	const defaultLearningRate = 0.001
	const defaultEpochs = 1
	if err := ner.train(tmpFile.Name(), defaultLearningRate, defaultEpochs); err != nil {
		return err
	}

	return nil
}

func (ner *NER) Save(path string, exportOnnx bool) error {
	modelPath := filepath.Join(path, "model.bin")
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	var err *C.char
	C.NER_save(ner.model, cPath, &err)
	if err != nil {
		defer C.free(unsafe.Pointer(err))
		return errors.New(C.GoString(err))
	}
	return nil
}

func newStringList(values []string) *C.StringList_t {
	list := C.StringList_new()
	for _, v := range values {
		vCStr := C.CString(v)
		C.StringList_append(list, vCStr)
		C.free(unsafe.Pointer(vCStr))
	}
	return list
}

func (ner *NER) Release() {
	if ner.model != nil {
		C.NER_free(ner.model)
		ner.model = nil
	}
}

func writeSamplesToCSV(w io.Writer, samples []api.Sample, tokensCol, tagsCol string) error {
	writer := csv.NewWriter(w)
	// header
	if err := writer.Write([]string{tokensCol, tagsCol}); err != nil {
		return err
	}
	// rows
	for _, sample := range samples {
		t := strings.Join(sample.Tokens, " ")
		l := strings.Join(sample.Labels, " ")
		if err := writer.Write([]string{t, l}); err != nil {
			return err
		}
	}
	writer.Flush()
	return writer.Error()
}
