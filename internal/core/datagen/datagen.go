package datagen

import (
	"errors"
	"fmt"
	"log/slog"
	"math/rand"
	"ner-backend/internal/core/utils"
	"ner-backend/pkg/api"
	"reflect"
	"regexp"
	"strings"

	"github.com/jaswdr/faker/v2"
	"github.com/openai/openai-go"
)

const maxConcurrentLLMCalls = 5

type DatagenOpts struct {
	Tags    []string
	Samples []api.Sample

	NumValuesPerTag    int // The number of values to generate for each tag
	RecordsToGenerate  int // The total number of Records to generate
	RecordsPerTemplate int // The number of Records to create from each template

	TestSplit float32
}

func templatizeSamples(tagsName []string, sample []api.Sample) ([]string, map[string][]string) {
	tagExamples := make(map[string][]string)

	// We should generate values for all tags, even if they are not present in any sample.
	for _, Name := range tagsName {
		tagExamples[Name] = make([]string, 0)
	}

	templatizeSingleSample := func(sample api.Sample) (string, map[string][]string) {
		template := make([]string, 0, len(sample.Tokens))
		sampleTagExamples := make(map[string][]string)

		for i, token := range sample.Tokens {
			if sample.Labels[i] != "O" {
				if i > 0 && sample.Labels[i] == sample.Labels[i-1] {
					sampleTagExamples[sample.Labels[i]][len(sampleTagExamples[sample.Labels[i]])-1] += " " + token
				} else {
					template = append(template, fmt.Sprintf("[%s]", sample.Labels[i]))
					sampleTagExamples[sample.Labels[i]] = append(sampleTagExamples[sample.Labels[i]], token)
				}
			} else {
				template = append(template, token)
			}
		}
		return strings.Join(template, " "), sampleTagExamples
	}

	templates := make([]string, 0, len(sample))
	for _, s := range sample {
		template, tagExamplesInSample := templatizeSingleSample(s)
		templates = append(templates, template)

		for tag, values := range tagExamplesInSample {
			tagExamples[tag] = append(tagExamples[tag], values...)
		}
	}

	// Remove duplicates from tagExamples
	for tag, examples := range tagExamples {
		seen := make(map[string]struct{})
		unique := make([]string, 0, len(examples))
		for _, v := range examples {
			if _, ok := seen[v]; !ok {
				seen[v] = struct{}{}
				unique = append(unique, v)
			}
		}
		tagExamples[tag] = unique
	}

	return templates, tagExamples
}

func GenerateData(opts DatagenOpts) ([]api.Sample, []api.Sample, error) {
	llm := NewOpenAI(openai.ChatModelGPT4oMini, 0.8)

	slog.Info("starting data generation", "tags", opts.Tags, "samples", len(opts.Samples), "numValuesPerTag", opts.NumValuesPerTag, "recordsToGenerate", opts.RecordsToGenerate, "recordsPerTemplate", opts.RecordsPerTemplate, "testSplit", opts.TestSplit)

	feedbackTemplates, tagFeedbackValues := templatizeSamples(opts.Samples)

	values, err := getTagValues(llm, tagFeedbackValues, opts.NumValuesPerTag, 15)
	if err != nil {
		slog.Error("error getting tag values", "error", err)
		return nil, nil, err
	}

	slog.Info("generated tag values")

	trainValues, testValues := splitTrainTestValues(values, opts.TestSplit)

	slog.Info("generated prompts from tags")

	totalTemplates := opts.RecordsToGenerate / opts.RecordsPerTemplate
	templateGenerationPrompts, err := createPromptsFromSamples(opts.Tags, feedbackTemplates, totalTemplates)
	if err != nil {
		slog.Error("error creating prompts from samples", "error", err)
		return nil, nil, err
	}

	slog.Info("generated prompts from samples")

	templates, err := generateTemplates(templateGenerationPrompts, llm)
	if err != nil {
		slog.Error("error generating templates", "error", err)
		return nil, nil, err
	}

	slog.Info("generated templates")

	trainTemplates, testTemplates := trainTestSplit(templates, opts.TestSplit)

	trainSamples, err := renderTemplates(trainTemplates, trainValues, opts.RecordsPerTemplate)
	if err != nil {
		slog.Error("error rendering train templates", "error", err)
		return nil, nil, err
	}

	slog.Info("rendered train templates")

	testSamples, err := renderTemplates(testTemplates, testValues, opts.RecordsPerTemplate)
	if err != nil {
		slog.Error("error rendering test templates", "error", err)
		return nil, nil, err
	}

	slog.Info("rendered test templates")

	slog.Info("generated data", "trainSamples", len(trainSamples), "testSamples", len(testSamples))

	return trainSamples, testSamples, nil
}

func getTagValues(llm LLM, tagsExamples map[string][]string, numValuesPerTag, generateAtOnce int) (map[string][]string, error) {
	faker := newFakerWrapper()

	values := make(map[string][]string)

	for tagName, feedbackTagValues := range tagsExamples {

		allValues := faker.getTagValues(tagName, numValuesPerTag*3/2)

		if len(feedbackTagValues) < 20 {
			// mixin faker values if we don't have enough feedback samples
			feedbackTagValues = append(feedbackTagValues, randomSample(allValues, min(len(allValues), 20))...)
		}
		llmValues, err := getTagValuesFromLLM(llm, tagName, feedbackTagValues, numValuesPerTag, generateAtOnce)
		if err != nil {
			return nil, fmt.Errorf("error getting values from llm for tag '%s': %w", tagName, err)
		}
		allValues = append(allValues, llmValues...)

		seen := make(map[string]struct{})
		unique := make([]string, 0, len(allValues))
		for _, v := range allValues {
			v := strings.TrimSpace(v)
			vLower := strings.ToLower(v)
			if _, duplicate := seen[vLower]; len(v) > 0 && !duplicate {
				seen[vLower] = struct{}{}
				unique = append(unique, v)
			}
		}

		values[tagName] = unique
	}

	return values, nil
}

type fakerWrapper struct {
	faker   faker.Faker
	methods map[string]func() string
}

func newFakerWrapper() *fakerWrapper {
	wrapper := &fakerWrapper{
		faker:   faker.New(),
		methods: make(map[string]func() string),
	}

	t := reflect.TypeOf(wrapper.faker)
	v := reflect.ValueOf(wrapper.faker)
	for i := range t.NumMethod() {
		method := t.Method(i)
		switch method.Name {
		case "Person", "Address", "Phone", "Company", "Time", "Internet", "Payment", "Currency":
			domain := v.MethodByName(method.Name).Call(nil)[0]
			domainType := domain.Type()
			for j := range domainType.NumMethod() {
				domainMethod := domainType.Method(j)
				f, ok := domain.MethodByName(domainMethod.Name).Interface().(func() string)
				if ok {
					wrapper.methods[strings.ToLower(domainMethod.Name)] = f
					wrapper.methods[strings.ToLower(method.Name+domainMethod.Name)] = f
				}
			}
		}
	}

	return wrapper
}

func randomSample[T any](input []T, k int) []T {
	if k > len(input) {
		panic(fmt.Sprintf("k (%d) cannot be greater than the length of input (%d)", k, len(input)))
	}

	indices := rand.Perm(len(input))[:k]

	result := make([]T, k)
	for i, idx := range indices {
		result[i] = input[idx]
	}
	return result
}

func (w *fakerWrapper) getTagValues(tag string, numValuesPerTag int) []string {
	cleanedTag := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(tag, "_", ""), "-", ""))

	if f, ok := w.methods[cleanedTag]; ok {
		values := make([]string, numValuesPerTag)
		for i := 0; i < numValuesPerTag; i++ {
			values[i] = f()
		}
		return values
	}
	return make([]string, 0)
}

func getTagValuesFromLLM(llm LLM, tagName string, examples []string, numValuesPerTag, generateAtOnce int) ([]string, error) {
	prompts := make([]string, 0, numValuesPerTag/generateAtOnce)
	for i := 0; i < numValuesPerTag; i += generateAtOnce {
		prompt := new(strings.Builder)
		err := tagValuePromptTmpl.Execute(prompt, tagValuePromptFields{
			NumValues: min(generateAtOnce, numValuesPerTag-i),
			Tag:       tagName,
			Examples:  randomSample(examples, min(6, len(examples))),
		})
		if err != nil {
			return nil, fmt.Errorf("error rendering tagValues template: %w", err)
		}
		prompts = append(prompts, prompt.String())
	}

	worker := func(prompt string) (string, error) {
		return llm.Generate("", prompt)
	}
	queue := make(chan string, len(prompts))
	for _, prompt := range prompts {
		queue <- prompt
	}
	close(queue)
	completed := make(chan utils.CompletedTask[string], len(prompts))

	utils.RunInPool(worker, queue, completed, maxConcurrentLLMCalls)

	var values []string
	var errs []error
	for response := range completed {
		if response.Error != nil {
			errs = append(errs, response.Error)
		} else {
			values = append(values, strings.Split(response.Result, "\n")...)
		}
	}

	if len(errs) > 0 {
		return values, errors.Join(errs[:min(3, len(errs))]...)
	}
	return values, nil
}

func trainTestSplit[T any](data []T, testSplit float32) ([]T, []T) {
	trainLen := int(float32(len(data)) * (1 - testSplit))

	// shuffle the data
	data = randomSample(data, len(data))

	return data[:trainLen], data[trainLen:]
}

// This function splits the sets of candidate values in separate train and test sets. This prevents the model
// from cheating the test set by memorizing the tags for a given token.
func splitTrainTestValues(values map[string][]string, testSplit float32) (map[string][]string, map[string][]string) {
	train := make(map[string][]string)
	test := make(map[string][]string)

	for k, v := range values {
		train[k], test[k] = trainTestSplit(v, testSplit)
	}

	return train, test
}

// This function creates a set of prompts that can be used to generate templates based one each user provided sample.
func createPromptsFromSamples(tags []string, exampleTemplates []string, totalTemplates int) ([]string, error) {
	var prompts []string

	templatesPerPrompt := 10

	for i := 0; i < totalTemplates; i += templatesPerPrompt {
		prompt := new(strings.Builder)

		// pick 3 random samples from the provided samples
		sampledExampleTemplates := randomSample(exampleTemplates, min(3, len(exampleTemplates)))

		err := templateFromSamplePromptTmpl.Execute(prompt, templateFromSamplePromptFields{
			NumTemplates: min(templatesPerPrompt, totalTemplates-i),
			TagsName:     tags,
			Samples:      sampledExampleTemplates,
		})
		if err != nil {
			return nil, fmt.Errorf("error rendering promptFromSample template: %w", err)
		}
		prompts = append(prompts, prompt.String())
	}

	return prompts, nil
}

// This function makes calls to the LLM to foreach template generation prompt and returns all of the generated templates.
func generateTemplates(prompts []string, llm LLM) ([]string, error) {
	systemPrompt := "You are a helpful assistant designed to generate synthetic data for NER (Named Entity Recognition) tasks. Your goal is to create diverse and realistic templates that can be used to generate samples for various named entities."

	worker := func(prompt string) (string, error) {
		return llm.Generate(systemPrompt, prompt)
	}

	queue := make(chan string, len(prompts))
	for _, prompt := range prompts {
		queue <- prompt
	}
	close(queue)
	completed := make(chan utils.CompletedTask[string], len(prompts))

	utils.RunInPool(worker, queue, completed, maxConcurrentLLMCalls)

	templates := make([]string, 0, len(prompts))
	var errs []error
	for response := range completed {
		if response.Error != nil {
			errs = append(errs, response.Error)
		} else {
			if len(response.Result) > 0 { // TODO: should this threshold be higher?
				templates = append(templates, response.Result)
			}
		}
	}

	if len(errs) > 0 {
		return templates, errors.Join(errs[:min(3, len(errs))]...)
	}
	return templates, nil
}

// This function takes in a set of templates and the sets of values foreach tag and generates a
// set of samples by replacing every tag placeholder in each template with a random value from
// the set of values for that tag.
func renderTemplates(templates []string, values map[string][]string, samplesPerTemplate int) ([]api.Sample, error) {
	samples := make([]api.Sample, 0, len(templates)*samplesPerTemplate)

	for _, template := range templates {
		renderedSamples, err := renderTemplate(template, values, samplesPerTemplate)
		if err != nil { // TODO: should we ignore errors here as long as most templates are rendered?
			return nil, err
		}
		samples = append(samples, renderedSamples...)
	}

	return samples, nil
}

var tagRe = regexp.MustCompile(`\[(\w+?)\]`)

func renderTemplate(template string, values map[string][]string, samplesPerTemplate int) ([]api.Sample, error) {
	samples := make([]api.Sample, samplesPerTemplate)

	tokens := strings.Fields(template)

	for _, token := range tokens {
		match := tagRe.FindStringSubmatch(token)
		if match != nil {
			tag := match[1]

			tagValues := values[tag]

			if len(tagValues) == 0 {
				return nil, fmt.Errorf("cannot render template '%s', no values found for tag '%s'", template, tag)
			}

			for i := range samples {
				valueTokens := strings.Fields(tagValues[rand.Intn(len(tagValues))])
				samples[i].Tokens = append(samples[i].Tokens, valueTokens...)

				for range len(valueTokens) {
					samples[i].Labels = append(samples[i].Labels, tag)
				}
			}
		} else {
			for i := range samples {
				samples[i].Tokens = append(samples[i].Tokens, token)
				samples[i].Labels = append(samples[i].Labels, "O")
			}
		}
	}

	return samples, nil
}
