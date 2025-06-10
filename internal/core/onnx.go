package core

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

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

type CRF struct {
	Transitions [][]float32
	StartProbs  []float32
	EndProbs    []float32
}

func loadCRF(path string) (CRF, error) {
	file, err := os.Open(path)
	if err != nil {
		return CRF{}, err
	}
	var data CRF
	if err := json.NewDecoder(file).Decode(&data); err != nil {
		return CRF{}, err
	}
	return data, nil
}

func (crf *CRF) NumTags() int {
	return len(crf.Transitions)
}

func (crf *CRF) ViterbiDecode(emissions [][]float32) []int {
	nTags := crf.NumTags()
	seqLen := len(emissions)
	dp := make([][]float32, seqLen)
	bp := make([][]int, seqLen)
	for t := 0; t < seqLen; t++ {
		dp[t] = make([]float32, nTags)
		bp[t] = make([]int, nTags)
	}
	for j := 0; j < nTags; j++ {
		dp[0][j] = emissions[0][j] + crf.StartProbs[j]
	}
	for t := 1; t < seqLen; t++ {
		for currTag := 0; currTag < nTags; currTag++ {
			bestScore := float32(-1e9)
			var bestPrevTag int
			for prevTag := 0; prevTag < nTags; prevTag++ {
				s := dp[t-1][prevTag] + crf.Transitions[prevTag][currTag] + emissions[t][currTag]
				if s > bestScore {
					bestScore = s
					bestPrevTag = prevTag
				}
			}
			dp[t][currTag] = bestScore
			bp[t][currTag] = bestPrevTag
		}
	}

	for j := 0; j < nTags; j++ {
		dp[seqLen-1][j] += crf.EndProbs[j]
	}

	seq := make([]int, seqLen)
	bestTag := 0
	bestScore := float32(-1e9)
	for j := 0; j < nTags; j++ {
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

func getWordIds(wordOffsets [][2]int, tokenOffsets []tokenizers.Offset) []int {
	// This function assumes that the word/token offets are non-overlapping and sorted.
	wordIDs := make([]int, len(tokenOffsets))
	wordID := 0

	for i, off := range tokenOffsets {
		tokenStart, tokenEnd := int(off[0]), int(off[1])

		for wordID < len(wordOffsets) && wordOffsets[wordID][1] <= tokenStart {
			wordID++ // skip words that end before the token starts
		}

		if wordID < len(wordOffsets) && wordOffsets[wordID][0] < tokenEnd {
			wordIDs[i] = wordID // token overlaps with this word
		} else {
			wordIDs[i] = -1 // token does not overlap with any word
		}
	}

	return wordIDs
}

func aggregatePredictions(tags []string, wordIds []int, numWords int) []string {
	preds := make([]string, numWords)
	for i := range preds {
		preds[i] = "O"
	}

	for i, tag := range tags {
		if wordID := wordIds[i]; wordID >= 0 && preds[wordID] == "O" {
			preds[wordID] = tag
		}
	}

	return preds
}

type OnnxModel struct {
	session   *ort.DynamicAdvancedSession
	tokenizer *tokenizers.Tokenizer
	crf       CRF
}

func decryptModel(encPath, keyB64 string) ([]byte, error) {
	key, err := base64.StdEncoding.DecodeString(keyB64)
	if err != nil {
		return nil, fmt.Errorf("invalid MODEL_KEY: %w", err)
	}
	if len(key) != 32 {
		return nil, fmt.Errorf("MODEL_KEY must be 32 bytes")
	}

	ct, err := os.ReadFile(encPath)
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
	keyB64 := "UuTl+ZEVxcUCJoXIDkePg49vS/GYjHa+Fd96kp8vG5E="
	onnxBytes, err := decryptModel(encPath, keyB64)
	if err != nil {
		return nil, fmt.Errorf("decrypt model: %w", err)
	}

	crf, err := loadCRF(crfPath)
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
		session:   session,
		tokenizer: tk,
		crf:       crf,
	}, nil
}

func (m *OnnxModel) Predict(text string) ([]types.Entity, error) {
	cleanedText, originalWordOffsets, cleanedWordOffsets := CleanTextWithSpans(text)

	enc := m.tokenizer.EncodeWithOptions(cleanedText, false, tokenizers.WithReturnAllAttributes())
	ids := make([]int64, len(enc.IDs))
	for i, v := range enc.IDs {
		ids[i] = int64(v)
	}

	B, L, N := int64(1), int64(len(ids)), int64(m.crf.NumTags())

	inT, err := ort.NewTensor(ort.NewShape(B, L), ids)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := inT.Destroy(); err != nil {
			slog.Error("failed to destroy input tensor", "error", err)
		}
	}()

	outT, err := ort.NewEmptyTensor[float32](ort.NewShape(B, L, N))
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := outT.Destroy(); err != nil {
			slog.Error("failed to destroy output tensor", "error", err)
		}
	}()

	if err := m.session.Run([]ort.Value{inT}, []ort.Value{outT}); err != nil {
		return nil, fmt.Errorf("session run error: %w", err)
	}

	flat := outT.GetData()
	seq := make([][]float32, L)
	for t := int64(0); t < L; t++ {
		start := t * N
		seq[t] = flat[start : start+N]
	}

	tagsIdx := m.crf.ViterbiDecode(seq)
	subTags := make([]string, len(tagsIdx))
	for i, tagID := range tagsIdx {
		subTags[i] = idx2tag[tagID]
	}

	wordIDs := getWordIds(cleanedWordOffsets, enc.Offsets)

	wordTags := aggregatePredictions(subTags, wordIDs, len(cleanedWordOffsets))

	var ents []types.Entity
	for wid, tag := range wordTags {
		if tag == "O" {
			continue
		}

		ents = append(ents, types.CreateEntity(
			tag,
			text,
			originalWordOffsets[wid][0],
			originalWordOffsets[wid][1],
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
	if err := m.session.Destroy(); err != nil {
		slog.Error("failed to destroy ONNX session", "error", err)
	}
	if err := m.tokenizer.Close(); err != nil {
		slog.Error("failed to close tokenizer", "error", err)
	}
}
