package thirdai_nlp

// #cgo linux LDFLAGS: -L./lib/linux_amd64 -L./lib/linux_arm64 -lthirdai_nlp_c
// #cgo darwin LDFLAGS: -L./lib/macos_arm64 -L /Users/nmeisburger/ThirdAI/libtorch/lib -lthirdai_nlp_c -ltorch -ltorch_cpu -lc10 -lstdc++ -lbz2 -Wl,-rpath,/Users/nmeisburger/ThirdAI/libtorch/lib
// #cgo CFLAGS: -O3
// #cgo CXXFLAGS: -O3 -fPIC -std=c++17 -I./include -fvisibility=hidden
// #include <stdlib.h>
// #include "thirdai_nlp.h"
import "C"
import (
	"fmt"

	"ner-backend/internal/core/types"
	"ner-backend/internal/core/utils"
	"ner-backend/pkg/api"
	"unsafe"
)

type CnnModel struct {
	ptr unsafe.Pointer
}

func LoadCnnModel(modelPath string) (*CnnModel, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	result := C.cnn_model_load(cModelPath)
	defer C.cnn_model_free_load_result(result)
	if result.error_msg != nil {
		return nil, fmt.Errorf("failed to load cnn model: %v", C.GoString(result.error_msg))
	}

	return &CnnModel{ptr: result.model_ptr}, nil
}

func (m *CnnModel) Release() {
	if m.ptr != nil {
		C.cnn_model_free(m.ptr)
		m.ptr = nil
	}
}

func (m *CnnModel) Predict(text string) ([]types.Entity, error) {
	batch, offsets := utils.SplitText(text)

	cBatch := make([]*C.char, len(batch))
	for i, s := range batch {
		cBatch[i] = C.CString(s)
	}
	defer func() {
		for _, s := range cBatch {
			C.free(unsafe.Pointer(s))
		}
	}()

	cResult := C.cnn_model_predict(m.ptr, &cBatch[0], C.ulong(len(batch)))
	defer C.free_predictions(cResult)

	if cResult.error_msg != nil {
		return nil, fmt.Errorf("cnn model inference failed: %v", C.GoString(cResult.error_msg))
	}

	results := make([]types.Entity, 0)

	batchOffsets := unsafe.Slice(cResult.batch_offsets, cResult.batch_size+1)
	predictions := unsafe.Slice(cResult.predictions, batchOffsets[len(batchOffsets)-1])

	for i := range cResult.batch_size {
		start := int(batchOffsets[i])
		end := int(batchOffsets[i+1])

		for _, pred := range predictions[start:end] {
			label := C.GoString(pred.label)

			if label == "GENDER" || label == "SEXUAL_ORENTATION" || label == "ENTNICITY" {
				continue
			}

			entity := types.CreateEntity(
				C.GoString(pred.label),
				text,
				int(pred.start)+offsets[i],
				int(pred.end)+offsets[i],
			)

			results = append(results, entity)
		}
	}

	return results, nil
}

func (m *CnnModel) Save(path string) error {
	return fmt.Errorf("save not implemented for CNN model")
}

func (m *CnnModel) Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error {
	return fmt.Errorf("finetune not implemented for CNN model")
}
