package datagen

import (
	"errors"
	"fmt"
	"math/rand"
	"ner-backend/internal/core/utils"
	"reflect"
	"regexp"
	"slices"
	"strings"

	"github.com/jaswdr/faker/v2"
	"github.com/openai/openai-go"
)

const maxConcurrentLLMCalls = 5

type DatagenOpts struct {
	TaskPrompt string
	Tags       []TagInfo
	Samples    []Sample

	NumValuesPerTag    int // The number of values to generate for each tag
	SamplesToGenerate  int // The total number of samples to create from templates
	SamplesPerTemplate int // The number of samples to create from each template
	GenerateAtOnce     int // The number of templates to generate with LLM call
	TemplatesPerSample int // The number of templates to generate for each provided sample

	TestSplit float32
}

func GenerateData(opts DatagenOpts) ([]Sample, []Sample, error) {
	llm := NewOpenAI(openai.ChatModelGPT4oMini, 0.8)

	values, err := getTagValues(llm, opts.TaskPrompt, opts.Tags, opts.NumValuesPerTag, opts.GenerateAtOnce)
	if err != nil {
		return nil, nil, err
	}

	trainValues, testValues := splitTrainTestValues(values, opts.TestSplit)

	tags, err := getExtendedDescriptions(opts.Tags, nil)
	if err != nil {
		return nil, nil, err
	}

	nTemplates := opts.SamplesToGenerate / opts.SamplesPerTemplate

	tagPrompts, err := createPromptsFromTags(opts.TaskPrompt, tags, nTemplates, opts.GenerateAtOnce)
	if err != nil {
		return nil, nil, err
	}

	samplePrompts, err := createPromptsFromSamples(opts.TaskPrompt, tags, opts.Samples, opts.TemplatesPerSample)
	if err != nil {
		return nil, nil, err
	}

	templates, err := generateTemplates(opts.TaskPrompt, slices.Concat(tagPrompts, samplePrompts), llm)
	if err != nil {
		return nil, nil, err
	}

	trainTemplates, testTemplates := trainTestSplit(templates, opts.TestSplit)

	trainSamples, err := renderTemplates(trainTemplates, trainValues, opts.SamplesPerTemplate)
	if err != nil {
		return nil, nil, err
	}

	testSamples, err := renderTemplates(testTemplates, testValues, opts.SamplesPerTemplate)
	if err != nil {
		return nil, nil, err
	}

	return trainSamples, testSamples, nil
}

func getTagValues(llm LLM, taskPrompt string, tags []TagInfo, numValuesPerTag, generateAtOnce int) (map[string][]string, error) {
	faker := newFakerWrapper()

	values := make(map[string][]string)

	for _, tag := range tags {
		allValues := tag.Examples

		fakerValues := faker.getTagValues(tag.Name, numValuesPerTag*3/2)

		if len(fakerValues) > 0 {
			allValues = append(allValues, fakerValues...)
		} else {
			llmValues, err := getTagValuesFromLLM(llm, taskPrompt, tag, numValuesPerTag, generateAtOnce)
			if err != nil {
				return nil, fmt.Errorf("error getting values from llm for tag '%s': %w", tag.Name, err)
			}
			allValues = append(allValues, llmValues...)
		}

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

		values[tag.Name] = unique
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
				}
			}
		}
	}

	return wrapper
}

func (w *fakerWrapper) getTagValues(tag string, numValuesPerTag int) []string {
	cleanedTag := strings.ToLower(strings.Replace(tag, "_", "", -1))

	if f, ok := w.methods[cleanedTag]; ok {
		values := make([]string, numValuesPerTag)
		for i := 0; i < numValuesPerTag; i++ {
			values[i] = f()
		}
		return values
	}
	return nil
}

func getTagValuesFromLLM(llm LLM, taskPrompt string, tag TagInfo, numValuesPerTag int, generateAtOnce int) ([]string, error) {
	prompts := make([]string, 0, numValuesPerTag/generateAtOnce)
	for i := 0; i < numValuesPerTag; i += generateAtOnce {
		prompt := new(strings.Builder)
		err := tagValuePromptTmpl.Execute(prompt, tagValuePromptFields{
			TaskPrompt:  taskPrompt,
			NumValues:   min(generateAtOnce, numValuesPerTag-i),
			Tag:         tag.Name,
			Description: tag.Description,
			Examples:    tag.Examples[:min(3, len(tag.Examples))],
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
	perm := rand.Perm(len(data))

	train := make([]T, 0, trainLen)
	test := make([]T, 0, len(data)-trainLen)

	for _, i := range perm[:trainLen] {
		train = append(train, data[i])
	}

	for _, i := range perm[trainLen:] {
		test = append(test, data[i])
	}

	return train, test
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

// This function generates a longer description of each tag using an LLM.
func getExtendedDescriptions(tags []TagInfo, llm LLM) ([]tagInfoExtendedDescription, error) {
	extendedTags := make([]tagInfoExtendedDescription, 0, len(tags))

	for i, tag := range tags {
		prompt := new(strings.Builder)
		err := extendedDescriptionPromptTmpl.Execute(prompt, extendedDescriptionPromptFields{
			Name:        tag.Name,
			Description: tag.Description,
			Examples:    tag.Examples[:min(2, len(tag.Examples))],
		})
		if err != nil {
			return nil, fmt.Errorf("error rendering extendedDescription template: %w", err)
		}

		response, err := llm.Generate("", prompt.String())
		if err != nil {
			return nil, fmt.Errorf("error generating extended description: %w", err)
		}

		extendedTags[i] = tagInfoExtendedDescription{
			Name:                tag.Name,
			Description:         tag.Description,
			ExtendedDescription: response,
			Examples:            tag.Examples,
		}
	}

	return extendedTags, nil
}

// This function creates a set of prompts that can be used to generate templates for each tag.
func createPromptsFromTags(taskPrompt string, tags []tagInfoExtendedDescription, nTemplates, generateAtOnce int) ([]string, error) {
	var prompts []string

	for i := 0; i < nTemplates; i += generateAtOnce {
		tagSubset := make([]tagInfoExtendedDescription, min(len(tags), 4))
		perm := rand.Perm(len(tags))
		for i := range tagSubset {
			tagSubset[i] = tags[perm[i]]
		}

		prompt := new(strings.Builder)

		err := templateFromTagPromptTmpl.Execute(prompt, templateFromTagPromptFields{
			TaskPrompt:   taskPrompt,
			NumTemplates: min(generateAtOnce, nTemplates-i),
			TagsInfo:     tagSubset,
		})
		if err != nil {
			return nil, fmt.Errorf("error rendering promptFromTag template: %w", err)
		}

		prompts = append(prompts, prompt.String())
	}

	return prompts, nil
}

// This function creates a set of prompts that can be used to generate templates based one each user provided sample.
func createPromptsFromSamples(taskPrompt string, tags []tagInfoExtendedDescription, samples []Sample, templatesPerSample int) ([]string, error) {
	var prompts []string

	for _, sample := range samples {
		prompt := new(strings.Builder)

		err := templateFromSamplePromptTmpl.Execute(prompt, templateFromSamplePromptFields{
			templateFromTagPromptFields: templateFromTagPromptFields{
				TaskPrompt:   taskPrompt,
				NumTemplates: templatesPerSample,
				TagsInfo:     tags,
			},
			Sample: sample.asTemplate(),
		})
		if err != nil {
			return nil, fmt.Errorf("error rendering promptFromSample template: %w", err)
		}

		prompts = append(prompts, prompt.String())
	}

	return prompts, nil
}

// This function makes calls to the LLM to foreach template generation prompt and returns all of the generated templates.
func generateTemplates(taskPrompt string, prompts []string, llm LLM) ([]string, error) {
	systemPrompt := "You are a helpful assistant designed to generate synthetic data for domain " + taskPrompt

	worker := func(prompt string) (string, error) {
		return llm.Generate(systemPrompt, prompt)
	}

	queue := make(chan string, len(prompts))
	for _, prompt := range prompts {
		queue <- prompt
	}
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
func renderTemplates(templates []string, values map[string][]string, samplesPerTemplate int) ([]Sample, error) {
	samples := make([]Sample, 0, len(templates)*samplesPerTemplate)

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

func renderTemplate(template string, values map[string][]string, samplesPerTemplate int) ([]Sample, error) {
	samples := make([]Sample, samplesPerTemplate)

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
