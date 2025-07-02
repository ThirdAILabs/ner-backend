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

	openai "github.com/openai/openai-go"
)

// TagInfo mirrors your Python TypedDict.
type TagInfo struct {
	Name     string   // maps to tag["name"]
	Desc     string   // maps to tag["desc"]
	Examples []string // maps to tag["examples"]
	Contexts []string // maps to tag["contexts"], may be nil initially
}

// DataFactory drives the end-to-end generation.
type DataFactory struct {
	OutDir string
	LLM    *OpenAILLM
}

// NewDataFactory creates the output directory and LLM wrapper.
func NewDataFactory(outDir string) (*DataFactory, error) {
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return nil, err
	}
	usagePath := filepath.Join(outDir, "usage.json")
	llm := NewOpenAILLM("gpt-4o", usagePath, 0.0)
	return &DataFactory{OutDir: outDir, LLM: llm}, nil
}

// ExtendDescription calls the LLM to get an expanded tag description.
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

// ExtendExamples calls the LLM to get additional examples.
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

// GetTagContext calls the LLM to get context labels for a tag.
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

// runAndCollect fires off multiple prompts (in parallel if desired)
// and returns the raw JSON replies.
func (d *DataFactory) runAndCollect(
	batch []string,
	systemPrompt string,
	responseFormat openai.ChatCompletionNewParamsResponseFormatUnion,
	parallel bool,
) ([]string, error) {
	out := make([]string, len(batch))
	errs := make([]error, len(batch))
	var wg sync.WaitGroup

	for i, userPrompt := range batch {
		call := func(i int, p string) {
			defer wg.Done()
			res, err := d.LLM.Generate(systemPrompt, p, responseFormat)
			out[i], errs[i] = res, err
		}
		if parallel {
			wg.Add(1)
			go call(i, userPrompt)
		} else {
			wg.Add(1)
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
	TagsInfo           []TagInfo    // built from api.TagInfo
	Samples            []api.Sample // initial seed samples (optional)
	RecordsToGenerate  int          // total sentences to generate
	RecordsPerTemplate int          // sentences per LLM call
	TestSplit          float32      // fraction to reserve for test
	UserInstructions   []string     // extra LLM instructions
	WriteBatchSize     int          // batch size for LLM calls
}

// Generate runs the full pipeline: enrich tags, then generate & write annotated data.
func (d *DataFactory) Generate(
	opts GenerateOptions,
) ([]api.Sample, []api.Sample, error) {
	// 1) Enrich tags with description, examples, and contexts
	tags := opts.TagsInfo
	for i := range tags {
		// Description
		ed, err := d.ExtendDescription(&tags[i])
		if err != nil {
			return nil, nil, fmt.Errorf("extend description: %w", err)
		}
		tags[i].Desc = ed

		// Examples
		exs, err := d.ExtendExamples(&tags[i], 20)
		if err != nil {
			return nil, nil, fmt.Errorf("extend examples: %w", err)
		}
		tags[i].Examples = append(tags[i].Examples, exs...)

		// Contexts
		if tags[i].Contexts == nil {
			ctxs, err := d.GetTagContext(&tags[i], 25)
			if err != nil {
				return nil, nil, fmt.Errorf("get contexts: %w", err)
			}
			tags[i].Contexts = ctxs
		}
	}

	// 2) Compute batch parameters
	total := opts.RecordsToGenerate
	perCall := opts.RecordsPerTemplate
	batchSize := opts.WriteBatchSize
	if batchSize <= 0 {
		batchSize = total
	}
	if perCall <= 0 {
		perCall = total
	}
	callsPerBatch := batchSize / perCall

	// 3) Prepare JSON-schema response format
	annotFmt, err := prompts.ToResponseFormatUnion(prompts.AnnotatedDataFormat)
	if err != nil {
		return nil, nil, fmt.Errorf("convert annotated_data format: %w", err)
	}

	// 4) Annotate in batches
	var allSentences []string
	for offset := 0; offset < total; offset += batchSize {
		n := perCall
		if rem := total - offset; rem < perCall {
			n = rem
		}

		// Build prompt batch
		batch := make([]string, callsPerBatch)
		for i := 0; i < callsPerBatch; i++ {
			var buf bytes.Buffer
			if err := prompts.AnnotatedDataTmpl.Execute(&buf, map[string]interface{}{
				"TagInfo":          tags,
				"K":                n,
				"Requirements":     Requirements,
				"UserInstructions": opts.UserInstructions,
			}); err != nil {
				return nil, nil, fmt.Errorf("render prompt: %w", err)
			}
			batch[i] = buf.String()
		}

		raws, err := d.runAndCollect(batch, prompts.AnnotatedDataSystem, annotFmt, true)
		if err != nil {
			return nil, nil, fmt.Errorf("batch generate: %w", err)
		}

		for _, r := range raws {
			var tmp struct {
				Sentences []string `json:"sentences"`
			}
			if jerr := json.Unmarshal([]byte(r), &tmp); jerr == nil {
				allSentences = append(allSentences, tmp.Sentences...)
			}
		}
	}

	// 5) Transform sentences to api.Sample
	allSamples := make([]api.Sample, 0, len(allSentences))
	for _, s := range allSentences {
		src, tgt := transformSentence(s, tags)
		toks := strings.Fields(src)
		lbls := strings.Fields(tgt)
		if len(toks) == len(lbls) {
			allSamples = append(allSamples, api.Sample{Tokens: toks, Labels: lbls})
		}
	}

	// 6) Split into train/test
	splitIdx := int(float32(len(allSamples)) * (1 - opts.TestSplit))
	if splitIdx < 0 {
		splitIdx = 0
	} else if splitIdx > len(allSamples) {
		splitIdx = len(allSamples)
	}
	train := allSamples[:splitIdx]
	test := allSamples[splitIdx:]

	return train, test, nil
}

var (
	// compile once
	annotRe = regexp.MustCompile(`(?P<before>[^\w\s#]*)#+(?P<entity>[^#]+?)#+(?P<tag>[A-Z_]+)#+(?P<after>[^\w\s#']*[\w']*)?`)
)

// transformSentence applies your Python‐style regex tagging to a single sentence.
func transformSentence(text string, tags []TagInfo) (string, string) {
	var (
		srcTokens []string
		tgtTokens []string
		last      = 0
	)
	matches := annotRe.FindAllStringSubmatchIndex(text, -1)

	for _, m := range matches {
		start, end := m[0], m[1]

		// prefix
		if last < start {
			pref := strings.Fields(text[last:start])
			for _, w := range pref {
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

	// suffix
	if last < len(text) {
		suf := strings.Fields(text[last:])
		for _, w := range suf {
			srcTokens = append(srcTokens, w)
			tgtTokens = append(tgtTokens, "O")
		}
	}

	return strings.Join(srcTokens, " "), strings.Join(tgtTokens, " ")
}
