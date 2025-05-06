package bolt

// #cgo linux LDFLAGS: -L./lib/linux_amd64 -L./lib/linux_arm64 -lthirdai -lrocksdb -lutf8proc -lspdlog -fopenmp
// #cgo darwin LDFLAGS: -L./lib/macos_arm64 -lthirdai -lrocksdb -lutf8proc -lspdlog -L/opt/homebrew/opt/libomp/lib/ -lomp
// #cgo CFLAGS: -O3
// #cgo CXXFLAGS: -O3 -fPIC -std=c++17 -I./include -fvisibility=hidden
// #include "binding.h"
// #include <stdlib.h>
import "C"
import (
	"errors"
	"fmt"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
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
	sentences := splitSentences(text)

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

		for j := C.uint(0); j < itemResultsLen; j++ {
			start := int(C.Results_start(cResults, i, j))
			end := int(C.Results_end(cResults, i, j))
			entity := types.Entity{
				Label:    C.GoString(C.Results_label(cResults, i, j)),
				Text:     C.GoString(C.Results_text(cResults, i, j)),
				Start:    start,
				End:      end,
				LContext: strings.ToValidUTF8(sentences[i][max(start-20, 0):start], ""),
				RContext: strings.ToValidUTF8(sentences[i][end:min(len(sentences[i]), end+20)], ""),
			}
			entity.UpdateContext(sentences[i])
			results = append(results, entity)
		}
	}

	return results, nil
}

func (ner *NER) Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error {
	return fmt.Errorf("Finetune not implemented")
}

func (ner *NER) Save(path string) error {
	return fmt.Errorf("save not implemented")
}

func splitSentences(text string) []string {
	tokens := strings.Fields(text)
	const sentLen = 100
	sentences := make([]string, 0, (len(tokens)+sentLen-1)/sentLen)

	for start := 0; start < len(tokens); start += sentLen {
		end := min(start+sentLen, len(tokens))
		sentences = append(sentences, strings.Join(tokens[start:end], " "))
	}
	return sentences
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
