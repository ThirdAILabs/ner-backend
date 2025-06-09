package core

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"unicode"

	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	initOnce sync.Once
	initErr  error
)

var idx2tag = []string{
	"ADDRESS", "CARD_NUMBER", "COMPANY", "CREDIT_SCORE", "DATE",
	"EMAIL", "ETHNICITY", "GENDER", "ID_NUMBER", "LICENSE_PLATE",
	"LOCATION", "NAME", "O", "PHONENUMBER", "SERVICE_CODE",
	"SEXUAL_ORIENTATION", "SSN", "URL", "VIN",
}

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

type OnnxModel struct {
	session     *ort.DynamicAdvancedSession
	tokenizer   *tokenizers.Tokenizer
	transitions [][]float32
}

func decryptModel(encPath, keyB64 string) ([]byte, error) {
	key, err := base64.StdEncoding.DecodeString(keyB64)
	if err != nil {
		return nil, fmt.Errorf("invalid MODEL_KEY: %w", err)
	}
	if len(key) != 32 {
		return nil, fmt.Errorf("MODEL_KEY must be 32 bytes")
	}

	ct, err := ioutil.ReadFile(encPath)
	if err != nil {
		return nil, fmt.Errorf("read encrypted model: %w", err)
	}
	if len(ct) < 12 {
		return nil, fmt.Errorf("ciphertext too short")
	}
	nonce, ciphertext := ct[:12], ct[12:]

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	pt, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("AES-GCM decrypt failed: %w", err)
	}
	return pt, nil
}

func LoadOnnxModel(modelDir string) (Model, error) {
	encPath := filepath.Join(modelDir, "model.onnx.enc")
	crfPath := filepath.Join(modelDir, "transitions.json")

	// decrypt the onnx bytes into memory
	keyB64 := "4g3SSWw2CssTRoeW+0UqVEZjzP/zCEJKIK+1bFE0fYs="
	if keyB64 == "" {
		return nil, fmt.Errorf("MODEL_KEY not set")
	}
	onnxBytes, err := decryptModel(encPath, keyB64)
	if err != nil {
		return nil, fmt.Errorf("decrypt model: %w", err)
	}

	trans, err := loadCRF(crfPath)
	if err != nil {
		return nil, fmt.Errorf("CRF load error: %w", err)
	}

	tk, err := tokenizers.FromPretrained("Qwen/Qwen2.5-0.5B")
	if err != nil {
		return nil, fmt.Errorf("tokenizer load: %w", err)
	}

	session, err := ort.NewDynamicAdvancedSessionWithONNXData(
		onnxBytes,
		[]string{"input_ids"},
		[]string{"emissions"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create in-memory session: %w", err)
	}

	return &OnnxModel{
		session:     session,
		tokenizer:   tk,
		transitions: trans,
	}, nil
}

func (m *OnnxModel) Predict(text string) ([]types.Entity, error) {
	cleanedText, originalSpans := CleanTextWithSpans(text)
	enc := m.tokenizer.EncodeWithOptions(cleanedText, false, tokenizers.WithReturnAllAttributes())
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
	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(B, L, N))
	if err != nil {
		return nil, err
	}
	defer outT.Destroy()
	if err := m.session.Run([]ort.Value{inT}, []ort.Value{outT}); err != nil {
		return nil, fmt.Errorf("session run error: %w", err)
	}
	flat := outT.GetData()
	seq := make([][]float32, L)
	for t := int64(0); t < L; t++ {
		start := t * N
		seq[t] = flat[start : start+N]
	}

	oIdx := -1
	for i, tag := range idx2tag {
		if tag == "O" {
			oIdx = i
			break
		}
	}

	if oIdx < 0 {
		return nil, fmt.Errorf("O tag not found in model tags")
	}

	for t := 0; t < int(L); t++ {
		seq[t][oIdx] *= 0.7
	}

	tagsIdx := viterbi(seq, m.transitions, int(L))
	subTags := make([]string, len(tagsIdx))
	for i, j := range tagsIdx {
		if j >= 0 && j < len(idx2tag) {
			subTags[i] = idx2tag[j]
		} else {
			subTags[i] = "O"
		}
	}
	offsets := enc.Offsets
	wordIDs := manualWordIDs(cleanedText, offsets)
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

	spans := make([][2]int, len(groups))
	for wid, offs := range groups {
		if len(offs) == 0 {
			spans[wid] = [2]int{0, 0}
		} else {
			spans[wid] = [2]int{
				int(offs[0][0]),
				int(offs[len(offs)-1][1]),
			}
		}
	}

	var ents []types.Entity
	for wid, offs := range groups {
		if len(offs) == 0 {
			continue
		}
		tag := wordTags[wid]
		if tag == "O" {
			continue
		}

		ents = append(ents, types.CreateEntity(
			tag,
			text,
			originalSpans[wid][0],
			originalSpans[wid][1],
		))
	}
	return ents, nil
}

func (m *OnnxModel) Finetune(_ string, _ []api.TagInfo, _ []api.Sample) error {
	return fmt.Errorf("finetune not supported for ONNX")
}

func (m *OnnxModel) Save(path string) error {
	return fmt.Errorf("save not supported for ONNX")
}

func (m *OnnxModel) Release() {
	m.session.Destroy()
	m.tokenizer.Close()
}
