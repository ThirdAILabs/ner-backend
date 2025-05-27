package core

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"unicode"

	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

var idx2tag = []string{
	"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE",
	"EMAIL", "ETHNICITY", "GENDER", "ID_NUMBER", "LICENSE_PLATE",
	"LOCATION", "NAME", "O", "PHONENUMBER", "SERVICE_CODE",
	"SEXUAL_ORIENTATION", "SSN", "URL", "VIN",
}

// loadCRF loads transition probabilities from a JSON file.
func loadCRF(path string) ([][]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var mat [][]float32
	if err := json.Unmarshal(data, &mat); err != nil {
		return nil, err
	}
	return mat, nil
}

// viterbi decodes the most likely tag sequence.
func viterbi(emissions [][]float32, transitions [][]float32, seqLen int) []int {
	N := len(transitions)
	dp := make([][]float32, seqLen)
	bp := make([][]int, seqLen)
	for t := 0; t < seqLen; t++ {
		dp[t] = make([]float32, N)
		bp[t] = make([]int, N)
	}
	for j := 0; j < N; j++ {
		dp[0][j] = emissions[0][j]
	}
	for t := 1; t < seqLen; t++ {
		for j := 0; j < N; j++ {
			maxScore := float32(-1e9)
			var maxPrev int
			for k := 0; k < N; k++ {
				s := dp[t-1][k] + transitions[k][j] + emissions[t][j]
				if s > maxScore {
					maxScore = s
					maxPrev = k
				}
			}
			dp[t][j] = maxScore
			bp[t][j] = maxPrev
		}
	}
	// backtrack
	seq := make([]int, seqLen)
	bestTag := 0
	bestScore := float32(-1e9)
	for j := 0; j < N; j++ {
		if dp[seqLen-1][j] > bestScore {
			bestScore = dp[seqLen-1][j]
			bestTag = j
		}
	}
	seq[seqLen-1] = bestTag
	for t := seqLen - 1; t > 0; t-- {
		seq[t-1] = bp[t][seq[t]]
	}
	return seq
}

// manualWordIDs maps each subword offset to a word index.
func manualWordIDs(text string, offsets []tokenizers.Offset) []int {
	wordIDs := make([]int, len(offsets))
	cur, lastEnd := -1, -1
	for i, off := range offsets {
		start, end := int(off[0]), int(off[1])
		if start == 0 && end == 0 {
			wordIDs[i] = -1
		} else {
			if (start == 0 || unicode.IsSpace(rune(text[start]))) && start >= lastEnd {
				cur++
			}
			wordIDs[i] = cur
		}
		lastEnd = end
	}
	return wordIDs
}

// aggregatePredictions reduces subword tags to word-level tags.
func aggregatePredictions(tags []string, lens []int) []string {
	a := make([]string, len(lens))
	ptr := 0
	for wi, l := range lens {
		best := "O"
		for j := 0; j < l; j++ {
			if tags[ptr+j] != "O" {
				best = tags[ptr+j]
				break
			}
		}
		a[wi] = best
		ptr += l
	}
	return a
}

// OnnxModel wraps a DynamicAdvancedSession for ONNX inference.
type OnnxModel struct {
	session     *ort.DynamicAdvancedSession
	tokenizer   *tokenizers.Tokenizer
	transitions [][]float32
}

// LoadOnnxModel initializes the ONNX runtime, tokenizer, and session.
func LoadOnnxModel(modelDir string) (Model, error) {
	onnxPath := filepath.Join(modelDir, "model.onnx")
	crfPath := filepath.Join(modelDir, "transitions.json")
	// load CRF transitions
	trans, err := loadCRF(crfPath)
	if err != nil {
		return nil, fmt.Errorf("CRF load error: %w", err)
	}
	// tokenizer
	tk, err := tokenizers.FromPretrained("Qwen/Qwen2.5-0.5B")
	if err != nil {
		return nil, fmt.Errorf("tokenizer load: %w", err)
	}
	//TODO: Pass the variable, rather than using from environment variable
	dylib := os.Getenv("ONNX_RUNTIME_DYLIB")

	// 2) fallback to a hard-coded .app install path, if nothing provided
	if dylib == "" {
		exe, err := os.Executable()
		if err != nil {
			log.Fatalf("cannot locate executable: %v", err)
		}
		// e.g. /Applications/PocketShield.app/.../bin/main
		contents := filepath.Dir(filepath.Dir(filepath.Dir(exe)))
		dylib = filepath.Join(contents, "Frameworks", "libonnxruntime.dylib")
	}

	// Tell ONNX Runtime exactly which .dylib to load:
	ort.SetSharedLibraryPath(dylib)

	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("failed to init ONNX Runtime: %v", err)
	}

	log.Printf("✔️  Loaded ONNX Runtime from %s", dylib)

	// dynamic session
	sess, err := ort.NewDynamicAdvancedSession(
		onnxPath,
		[]string{"input_ids"},
		[]string{"emissions"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create session error: %w", err)
	}
	return &OnnxModel{session: sess, tokenizer: tk, transitions: trans}, nil
}

// Predict runs tokenization, ONNX inference, Viterbi decoding, and aggregation.
func (m *OnnxModel) Predict(text string) ([]types.Entity, error) {
	enc := m.tokenizer.EncodeWithOptions(text, false, tokenizers.WithReturnAllAttributes())
	// input tensor
	ids := make([]int64, len(enc.IDs))
	for i, v := range enc.IDs {
		ids[i] = int64(v)
	}
	B, L, N := int64(1), int64(len(ids)), int64(len(m.transitions))
	inT, err := ort.NewTensor(ort.NewShape(B, L), ids)
	if err != nil {
		return nil, err
	}
	defer inT.Destroy()
	// output tensor
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(B, L, N))
	if err != nil {
		return nil, err
	}
	defer outT.Destroy()
	// run
	if err := m.session.Run([]ort.Value{inT}, []ort.Value{outT}); err != nil {
		return nil, fmt.Errorf("session run error: %w", err)
	}
	// reshape emissions
	flat := outT.GetData()
	seq := make([][]float32, L)
	for t := int64(0); t < L; t++ {
		start := t * N
		seq[t] = flat[start : start+N]
	}
	// decode
	tagsIdx := viterbi(seq, m.transitions, int(L))
	subTags := make([]string, len(tagsIdx))
	for i, j := range tagsIdx {
		if j >= 0 && j < len(idx2tag) {
			subTags[i] = idx2tag[j]
		} else {
			subTags[i] = "O"
		}
	}
	// aggregate
	offsets := enc.Offsets
	wordIDs := manualWordIDs(text, offsets)
	maxW := 0
	for _, w := range wordIDs {
		if w > maxW {
			maxW = w
		}
	}
	sublens := make([]int, maxW+1)
	for _, w := range wordIDs {
		if w >= 0 {
			sublens[w]++
		}
	}
	wordTags := aggregatePredictions(subTags, sublens)

	maxW = len(sublens)
	groups := make([][]tokenizers.Offset, maxW)
	for subIdx, wid := range wordIDs {
		if wid >= 0 {
			groups[wid] = append(groups[wid], enc.Offsets[subIdx])
		}
	}

	// 2) for each wordID, compute span and tag
	var ents []types.Entity
	for wid, offs := range groups {
		if len(offs) == 0 {
			// no real subwords for this group—skip
			continue
		}
		// offsets are in-order, so:
		start := int(offs[0][0])
		end := int(offs[len(offs)-1][1])
		tag := wordTags[wid]
		ents = append(ents, types.CreateEntity(
			tag,
			text,
			start,
			end,
		))
	}
	return ents, nil
}

// Finetune not supported.
func (m *OnnxModel) Finetune(_ string, _ []api.TagInfo, _ []api.Sample) error {
	return fmt.Errorf("finetune not supported for ONNX")
}

// Save not supported.
func (m *OnnxModel) Save(path string) error {
	return fmt.Errorf("save not supported for ONNX")
}

// Release frees the session, tokenizer, and environment.
func (m *OnnxModel) Release() {
	m.session.Destroy()
	m.tokenizer.Close()
	ort.DestroyEnvironment()
}
