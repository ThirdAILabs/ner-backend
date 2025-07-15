package datagenv2

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"ner-backend/internal/core/datagen_v2/prompts"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"math/rand"

	"github.com/schollz/progressbar/v3"

	openai "github.com/openai/openai-go"
)

type DataFactory struct {
	OutDir string
	LLM    *OpenAILLM
}

func NewDataFactory(outDir string) (*DataFactory, error) {
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return nil, err
	}
	usagePath := filepath.Join(outDir, "usage.json")
	llm := NewOpenAILLM("gpt-4o", usagePath, 0.0)
	return &DataFactory{OutDir: outDir, LLM: llm}, nil
}

func (d *DataFactory) ExtendDescription(tag *types.TagInfo) (string, error) {
	var buf bytes.Buffer
	if err := prompts.ExtendDescriptionTmpl.Execute(&buf, map[string]interface{}{
		"Tag": tag,
	}); err != nil {
		return "", err
	}

	descriptionFormat, err := prompts.ToResponseFormatUnion(prompts.ExtendDescriptionFormat)
	if err != nil {
		return "", fmt.Errorf("convert extend_description format: %w", err)
	}

	raw, err := d.LLM.Generate(prompts.ExtendDescriptionSystem, buf.String(), descriptionFormat)
	if err != nil {
		return "", err
	}

	var resp struct {
		ExtendedDescription string `json:"extended_description"`
	}
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		return "", fmt.Errorf("unmarshal extend_description: %w", err)
	}
	return resp.ExtendedDescription, nil
}

func (d *DataFactory) ExtendExamples(tag *types.TagInfo, k int) ([]string, error) {
	var buf bytes.Buffer
	if err := prompts.ExtendExamplesTmpl.Execute(&buf, map[string]interface{}{
		"Tag": tag, "K": k,
	}); err != nil {
		return nil, err
	}

	examplesFormat, err := prompts.ToResponseFormatUnion(prompts.ExtendExamplesFormat)
	if err != nil {
		return nil, fmt.Errorf("convert extend_examples format: %w", err)
	}

	raw, err := d.LLM.Generate(prompts.ExtendExamplesSystem, buf.String(), examplesFormat)
	if err != nil {
		return nil, err
	}

	var resp struct {
		ExtendedExamples []string `json:"extended_examples"`
	}
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		return nil, fmt.Errorf("unmarshal extend_examples: %w", err)
	}
	return resp.ExtendedExamples, nil
}

func (d *DataFactory) GetTagContext(tag *types.TagInfo, k int) ([]string, error) {
	var buf bytes.Buffer
	if err := prompts.ContextTmpl.Execute(&buf, map[string]interface{}{
		"Tag": tag, "K": k,
	}); err != nil {
		return nil, err
	}

	contextFormat, err := prompts.ToResponseFormatUnion(prompts.ContextFormat)
	if err != nil {
		return nil, fmt.Errorf("convert context format: %w", err)
	}

	raw, err := d.LLM.Generate(prompts.ContextSystem, buf.String(), contextFormat)
	if err != nil {
		return nil, err
	}

	var resp struct {
		Scenarios []string `json:"scenarios"`
	}
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		return nil, fmt.Errorf("unmarshal context: %w", err)
	}
	return resp.Scenarios, nil
}

func (d *DataFactory) runAndCollect(
	batch []string,
	systemPrompt string,
	responseFormat openai.ChatCompletionNewParamsResponseFormatUnion,
	parallel bool,
) ([]string, error) {
	bar := progressbar.NewOptions(len(batch),
		progressbar.OptionSetDescription("⏳ processing"),
		progressbar.OptionClearOnFinish(),
	)
	out := make([]string, len(batch))
	errs := make([]error, len(batch))
	var wg sync.WaitGroup

	for i, userPrompt := range batch {
		call := func(i int, p string) {
			defer wg.Done()
			res, err := d.LLM.Generate(systemPrompt, p, responseFormat)
			out[i], errs[i] = res, err
			_ = bar.Add(1)
		}
		wg.Add(1)
		if parallel {
			go call(i, userPrompt)
		} else {
			call(i, userPrompt)
		}
	}
	wg.Wait()

	for _, e := range errs {
		if e != nil {
			if errors.Is(e, context.DeadlineExceeded) {
				// Ignore timeout errors
				continue
			}
			return nil, fmt.Errorf("batch error: %v", errs)
		}
	}
	return out, nil
}

type GenerateOptions struct {
	TagsInfo            []types.TagInfo // built from api.types.TagInfo
	Samples             []api.Sample    // initial seed samples (optional)
	RecordsToGenerate   int             // total sentences to generate
	RecordsPerLlmCall   int             // sentences per LLM call
	TestSplit           float32         // fraction to reserve for test
	UserInstructions    []string        // extra LLM instructions
	VerifyGeneratedData bool            // whether to verify generated data
}

func (opts *GenerateOptions) Validate() error {
	if opts.RecordsToGenerate <= 0 {
		return fmt.Errorf("invalid RecordsToGenerate: %d", opts.RecordsToGenerate)
	}
	if opts.RecordsPerLlmCall <= 0 {
		return fmt.Errorf("invalid RecordsPerLlmCall: %d", opts.RecordsPerLlmCall)
	}
	if opts.TestSplit < 0.0 || opts.TestSplit > 1.0 {
		return fmt.Errorf("invalid TestSplit: %f", opts.TestSplit)
	}
	return nil
}

func mergeWithCorrections(orig, cleaned []string) []string {
	n, m := len(orig), len(cleaned)
	merged := make([]string, 0, n)
	used := make([]bool, m)
	const threshold = 0.7

	for _, o := range orig {
		bestIdx, bestScore := -1, 0.0

		for i, c := range cleaned {
			if used[i] {
				continue
			}
			_, score := FindMostSimilar(c, []string{o})
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}

		if bestIdx >= 0 && bestScore > threshold {
			merged = append(merged, cleaned[bestIdx])
			used[bestIdx] = true
		} else {
			merged = append(merged, o)
		}
	}

	return merged
}

func (d *DataFactory) verifyGeneratedData(toVerify [][]string, tagsInfo []types.TagInfo) ([]string, error) {
	verifyFmt, err := prompts.ToResponseFormatUnion(prompts.AnnotatedTextSamplesFormat)
	if err != nil {
		return nil, fmt.Errorf("convert verification format: %w", err)
	}
	verifyPrompts := make([]string, len(toVerify))
	for i, orig := range toVerify {
		var buf bytes.Buffer
		if err := prompts.AnnotatedTextSamplesTmpl.Execute(&buf, map[string]interface{}{
			"TagInfo":        tagsInfo,
			"AnnotatedTexts": orig,
		}); err != nil {
			return nil, fmt.Errorf("render verify prompt: %w", err)
		}
		verifyPrompts[i] = buf.String()
	}

	rawVerify, err := d.runAndCollect(verifyPrompts, prompts.AnnotatedTextCorrectionSystem, verifyFmt, true)
	if err != nil {
		return nil, fmt.Errorf("batch verify: %w", err)
	}

	var wg sync.WaitGroup
	verified := make([][]string, len(toVerify))

	for i, orig := range toVerify {
		wg.Add(1)
		go func(idx int, original []string) {
			defer wg.Done()

			var ats prompts.AnnotatedTextSamples
			var cleaned []string
			if err := json.Unmarshal([]byte(rawVerify[idx]), &ats); err == nil {
				// this openai response adheres to the expected format, otherwise this batch verification will be skipped and the original will be used
				cleaned = ats.Clean().AnnotatedTexts
			}

			if len(cleaned) == 0 {
				verified[idx] = original // no cleaning, keep original
			} else {
				verified[idx] = mergeWithCorrections(original, cleaned)
			}
		}(i, orig)
	}

	wg.Wait()

	// Flatten the results
	var result []string
	for _, v := range verified {
		result = append(result, v...)
	}

	return result, nil
}

func (d *DataFactory) Generate(opts GenerateOptions) ([]api.Sample, []api.Sample, error) {
	if err := opts.Validate(); err != nil {
		return nil, nil, fmt.Errorf("invalid options: %w", err)
	}
	total := opts.RecordsToGenerate
	perCall := min(opts.RecordsPerLlmCall, total)
	batchSize := min(300, total)

	feedback := SamplesToAnnotatedStrings(opts.Samples)

	bar := progressbar.NewOptions(len(opts.TagsInfo),
		progressbar.OptionSetDescription("⏳ Enhancing tags"),
	)
	defer bar.Close()
	for i := range opts.TagsInfo {
		tag := &opts.TagsInfo[i]
		if tag.Name == "O" {
			continue
		}

		if len(strings.Fields(tag.Desc)) <= 10 {
			ed, err := d.ExtendDescription(tag)
			if err != nil {
				return nil, nil, fmt.Errorf("extend description: %w", err)
			}
			tag.Desc = ed
		}

		if len(tag.Examples) < 4 {
			exs, err := d.ExtendExamples(tag, 20)
			if err != nil {
				return nil, nil, fmt.Errorf("extend examples: %w", err)
			}
			tag.Examples = append(tag.Examples, exs...)
		}

		if len(opts.Samples) > 0 {
			// opts.Samples are used as context for generation
			tag.Contexts = make([]string, 0)
		} else if len(tag.Contexts) < 4 {
			ctxs, err := d.GetTagContext(tag, rand.Intn(11)+20)
			if err != nil {
				return nil, nil, fmt.Errorf("get contexts: %w", err)
			}
			tag.Contexts = ctxs
		}
		_ = bar.Add(1)
	}

	callsPerBatch := batchSize / perCall
	if callsPerBatch == 0 {
		callsPerBatch = 1
	}

	genFmt, err := prompts.ToResponseFormatUnion(prompts.AnnotatedDataFormat)
	if err != nil {
		return nil, nil, fmt.Errorf("convert generation format: %w", err)
	}

	var allSentences []string
	bar = progressbar.NewOptions((total+batchSize-1)/batchSize,
		progressbar.OptionSetDescription("⏳ Generating sentences"),
	)
	for offset := 0; offset < total; offset += batchSize {
		n := perCall
		if rem := total - offset; rem < perCall {
			n = rem
		}

		genPrompts := make([]string, callsPerBatch)
		for i := 0; i < callsPerBatch; i++ {
			var buf bytes.Buffer
			var reqs []string
			if len(opts.Samples) > 0 {
				reqs = ContextualExampleRequirements
			} else {
				reqs = append([]string{}, RequiredRequirements...)
				reqs = append(reqs, prompts.RandomSample(AdditionalRequirements, 4)...)
			}
			if err := prompts.AnnotatedDataTmpl.Execute(&buf, map[string]interface{}{
				"K":                n,
				"TagInfo":          opts.TagsInfo,
				"Requirements":     reqs,
				"UserInstructions": opts.UserInstructions,
				"Feedback":         feedback,
			}); err != nil {
				return nil, nil, fmt.Errorf("render gen prompt: %w", err)
			}
			genPrompts[i] = buf.String()
		}

		rawGen, err := d.runAndCollect(genPrompts, prompts.AnnotatedDataSystem, genFmt, true)
		if err != nil {
			return nil, nil, fmt.Errorf("batch generate: %w", err)
		}

		var toVerify [][]string
		for _, r := range rawGen {
			var ad prompts.AnnotatedData
			if err := json.Unmarshal([]byte(r), &ad); err != nil {
				// this openai response did not match expected format, so skip this batch (original sentences) entirely.
				continue
			}
			cleaned := ad.Clean().Sentences
			if len(cleaned) > 0 {
				toVerify = append(toVerify, cleaned)
			}
		}

		if opts.VerifyGeneratedData {
			verified, err := d.verifyGeneratedData(toVerify, opts.TagsInfo)
			if err != nil {
				return nil, nil, fmt.Errorf("verify generated data: %w", err)
			}
			allSentences = append(allSentences, verified...)
		} else {
			for _, recordsToVerify := range toVerify {
				allSentences = append(allSentences, recordsToVerify...)
			}
		}
		_ = bar.Add(1)
	}

	var allSamples []api.Sample
	for _, s := range allSentences {
		src, tgt, err := d.transformSentence(s, opts.TagsInfo)
		if err != nil {
			continue
		}
		toks := strings.Fields(src)
		lbls := strings.Fields(tgt)
		allSamples = append(allSamples, api.Sample{Tokens: toks, Labels: lbls})
	}

	split := int(float32(len(allSamples)) * (1 - opts.TestSplit))
	return allSamples[:split], allSamples[split:], nil
}

func (d *DataFactory) ValidateSentence(src, tgt string, tags []types.TagInfo) error {
	srcToks := strings.Fields(src)
	tgtToks := strings.Fields(tgt)
	if len(srcToks) != len(tgtToks) {
		return fmt.Errorf("token count mismatch %d vs %d", len(srcToks), len(tgtToks))
	}
	if strings.Contains(src, "#") {
		return fmt.Errorf("source contains invalid '#' character")
	}
	allO := true
	for _, t := range tgtToks {
		if t != "O" {
			allO = false
			break
		}
	}
	if allO {
		return fmt.Errorf("all target tokens are 'O'")
	}
	return nil
}

var (
	annotRe = regexp.MustCompile(`(?P<before>[^\s#])?#+(?P<entity>[^#]+?)#+(?P<tag>[A-Z_]+)#+(?P<after>[^\s#])?`)
)

func (d *DataFactory) transformSentence(
	text string,
	tags []types.TagInfo,
) (string, string, error) {
	var srcTokens, tgtTokens []string
	last := 0
	matches := annotRe.FindAllStringSubmatchIndex(text, -1)

	for _, m := range matches {
		start, end := m[0], m[1]

		if last < start {
			for _, w := range strings.Fields(text[last:start]) {
				srcTokens = append(srcTokens, w)
				tgtTokens = append(tgtTokens, "O")
			}
		}
		before := ""
		if m[2] >= 0 {
			before = text[m[2]:m[3]]
		}
		entity := strings.TrimSpace(text[m[4]:m[5]])
		tag := text[m[6]:m[7]]
		after := ""
		if m[8] >= 0 {
			after = text[m[8]:m[9]]
		}
		toks := strings.Fields(before + entity + after)

		mark := "O"
		for _, t := range tags {
			if t.Name == tag {
				mark = tag
				break
			}
		}
		for _, w := range toks {
			srcTokens = append(srcTokens, w)
			tgtTokens = append(tgtTokens, mark)
		}
		last = end
	}

	if last < len(text) {
		for _, w := range strings.Fields(text[last:]) {
			srcTokens = append(srcTokens, w)
			tgtTokens = append(tgtTokens, "O")
		}
	}

	src := strings.Join(srcTokens, " ")
	tgt := strings.Join(tgtTokens, " ")

	if err := d.ValidateSentence(src, tgt, tags); err != nil {
		return src, tgt, fmt.Errorf("validation failed for '%s': %w", text, err)
	}

	return src, tgt, nil
}
