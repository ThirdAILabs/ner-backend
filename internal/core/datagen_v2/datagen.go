package datagenv2

import (
	"bytes"
	"encoding/json"
	"fmt"
	"ner-backend/internal/core/datagen_v2/prompts"
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

type TagInfo struct {
	Name     string   // maps to tag["name"]
	Desc     string   // maps to tag["desc"]
	Examples []string // maps to tag["examples"]
	Contexts []string // maps to tag["contexts"], may be nil initially
}

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

func (d *DataFactory) ExtendDescription(tag *TagInfo) (string, error) {
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

func (d *DataFactory) ExtendExamples(tag *TagInfo, k int) ([]string, error) {
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

func (d *DataFactory) GetTagContext(tag *TagInfo, k int) ([]string, error) {
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
		progressbar.OptionSetWidth(30),
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
			return out, fmt.Errorf("batch error: %v", errs)
		}
	}
	return out, nil
}

type GenerateOptions struct {
	TagsInfo          []TagInfo    // built from api.TagInfo
	Samples           []api.Sample // initial seed samples (optional)
	RecordsToGenerate int          // total sentences to generate
	RecordsPerLlmCall int          // sentences per LLM call
	TestSplit         float32      // fraction to reserve for test
	UserInstructions  []string     // extra LLM instructions
	WriteBatchSize    int          // batch size for LLM calls
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

func (d *DataFactory) Generate(opts GenerateOptions) ([]api.Sample, []api.Sample, error) {
	total := opts.RecordsToGenerate
	if total < 0 {
		return nil, nil, fmt.Errorf("invalid RecordsToGenerate: %d", total)
	}
	perCall := opts.RecordsPerLlmCall
	if perCall < 0 {
		return nil, nil, fmt.Errorf("invalid RecordsPerLlmCall: %d", perCall)
	}
	batchSize := opts.WriteBatchSize
	if batchSize < 0 {
		return nil, nil, fmt.Errorf("invalid WriteBatchSize: %d", batchSize)
	}
	if opts.TestSplit < 0.0 || opts.TestSplit > 1.0 {
		return nil, nil, fmt.Errorf("invalid TestSplit: %f", opts.TestSplit)
	}

	if perCall > total {
		perCall = total
	}
	if batchSize > total {
		batchSize = total
	}

	feedback := SamplesToAnnotatedStrings(opts.Samples)

	for i := range opts.TagsInfo {
		tag := &opts.TagsInfo[i]

		ed, err := d.ExtendDescription(tag)
		if err != nil {
			return nil, nil, fmt.Errorf("extend description: %w", err)
		}
		tag.Desc = ed

		exs, err := d.ExtendExamples(tag, 20)
		if err != nil {
			return nil, nil, fmt.Errorf("extend examples: %w", err)
		}
		tag.Examples = append(tag.Examples, exs...)

		if tag.Contexts == nil {
			if len(opts.Samples) > 0 {
				tag.Contexts = feedback
			} else {
				ctxs, err := d.GetTagContext(tag, rand.Intn(11)+20)
				if err != nil {
					return nil, nil, fmt.Errorf("get contexts: %w", err)
				}
				tag.Contexts = ctxs
			}
		}
	}

	callsPerBatch := batchSize / perCall
	if callsPerBatch == 0 {
		callsPerBatch = 1
	}

	genFmt, err := prompts.ToResponseFormatUnion(prompts.AnnotatedDataFormat)
	if err != nil {
		return nil, nil, fmt.Errorf("convert generation format: %w", err)
	}
	verifFmt, err := prompts.ToResponseFormatUnion(prompts.AnnotatedTextSamplesFormat)
	if err != nil {
		return nil, nil, fmt.Errorf("convert verification format: %w", err)
	}

	var allSentences []string
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
				continue
			}
			cleaned := ad.Clean().Sentences
			if len(cleaned) > 0 {
				toVerify = append(toVerify, cleaned)
			}
		}

		verifPrompts := make([]string, len(toVerify))
		for i, sentences := range toVerify {
			var buf bytes.Buffer
			if err := prompts.AnnotatedTextSamplesTmpl.Execute(&buf, map[string]interface{}{
				"TagInfo":        opts.TagsInfo,
				"AnnotatedTexts": sentences,
			}); err != nil {
				return nil, nil, fmt.Errorf("render verif prompt: %w", err)
			}
			verifPrompts[i] = buf.String()
		}
		rawVerif, err := d.runAndCollect(verifPrompts, prompts.AnnotatedTextCorrectionSystem, verifFmt, true)
		if err != nil {
			return nil, nil, fmt.Errorf("batch verify: %w", err)
		}

		for i, orig := range toVerify {
			var ats prompts.AnnotatedTextSamples
			if err := json.Unmarshal([]byte(rawVerif[i]), &ats); err != nil {
				return nil, nil, fmt.Errorf("unmarshal verification response: %w", err)
			}
			cleaned := ats.Clean().AnnotatedTexts
			if len(cleaned) == 0 {
				allSentences = append(allSentences, orig...)
			} else {
				allSentences = append(allSentences, mergeWithCorrections(orig, cleaned)...)
			}
		}
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

var (
	annotRe = regexp.MustCompile(`(?P<before>[^\w\s#]*)#+(?P<entity>[^#]+?)#+(?P<tag>[A-Z_]+)#+(?P<after>[^\w\s#']*[\w']*)?`)
)

func (d *DataFactory) ValidateSentence(src, tgt string, tags []TagInfo) error {
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

func (d *DataFactory) transformSentence(
	text string,
	tags []TagInfo,
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
		entity := strings.TrimSpace(text[m[2]:m[3]])
		tag := strings.ToUpper(text[m[4]:m[5]])
		toks := strings.Fields(entity)

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
