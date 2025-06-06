package core

import (
	_ "embed"
	"fmt"
	"ner-backend/internal/core/types"
	"ner-backend/pkg/api"
	"regexp"
	"strings"

	"gopkg.in/yaml.v2"
)

type RecognizerResult struct {
	EntityType string
	Match      string
	Score      float64
	Start, End int
}

type PatternRegex struct {
	Regex *regexp.Regexp
	Score float64
}

type PatternRecognizer struct {
	EntityType string
	Regexps    []PatternRegex
	Validate   func(string) bool
}

var entitiesMap = map[string]string{
	"UsLicenseRecognizer":             "VIN", // AKA US_DRIVER_LICENSE
	"DateRecognizer":                  "DATE",
	"EmailRecognizer":                 "EMAIL",
	"CreditCardRecognizer":            "CARD_NUMBER",
	"UsSsnRecognizer":                 "SSN",
	"UrlRecognizer":                   "URL",
	"UsPassportRecognizer":            "ID_NUMBER",
	"UsItinRecognizer":                "ID_NUMBER",
	"UsBankRecognizer":                "ID_NUMBER",
	"InPanRecognizer":                 "ID_NUMBER",
	"InAadhaarRecognizer":             "ID_NUMBER",
	"InVehicleRegistrationRecognizer": "VIN",
}

//go:embed recognizers.yaml
var recognizersYAML []byte

func loadPatterns() ([]*PatternRecognizer, error) {
	raw := struct {
		Recognizers []struct {
			Name     string `yaml:"name"`
			Patterns []struct {
				Regex string  `yaml:"regex"`
				Score float64 `yaml:"score,omitempty"`
			} `yaml:"patterns"`
		} `yaml:"recognizers"`
	}{}

	if err := yaml.Unmarshal(recognizersYAML, &raw); err != nil {
		return nil, err
	}

	var out []*PatternRecognizer
	for _, rec := range raw.Recognizers {
		pr := &PatternRecognizer{
			EntityType: rec.Name,
			Validate:   nil,
		}
		for _, p := range rec.Patterns {
			rxText := p.Regex

			// PAN (InPanRecognizer) low-strength lookahead
			if rec.Name == "InPanRecognizer" && strings.Contains(rxText, "(?=") {
				base := `\b[\w@#$%^?~-]{10}\b`
				rx := regexp.MustCompile(base)
				pr.Regexps = append(pr.Regexps, PatternRegex{
					Regex: rx,
					Score: p.Score,
				})
				pr.Validate = func(s string) bool {
					letters, digits := 0, 0
					for _, r := range s {
						switch {
						case '0' <= r && r <= '9':
							digits++
						case ('a' <= r && r <= 'z') || ('A' <= r && r <= 'Z'):
							letters++
						}
					}
					return letters >= 1 && digits >= 4
				}
				continue
			}

			// Vehicle reg part 1: (I)(?!00000)\d{5}
			if rec.Name == "InVehicleRegistrationRecognizer" && strings.Contains(rxText, "(?!00000)") {
				base := `\bI[0-9]{5}\b`
				rx := regexp.MustCompile(base)
				pr.Regexps = append(pr.Regexps, PatternRegex{
					Regex: rx,
					Score: p.Score,
				})
				pr.Validate = func(s string) bool {
					return s[1:] != "00000"
				}
				continue
			}

			// Vehicle reg part 2: (?!00)\d{2}[A-FH-KPRX]\d{6}[A-Z]
			if rec.Name == "InVehicleRegistrationRecognizer" && strings.Contains(rxText, "(?!00)") {
				base := `\b[0-9]{2}[A-FH-KPRX][0-9]{6}[A-Z]\b`
				rx := regexp.MustCompile(base)
				pr.Regexps = append(pr.Regexps, PatternRegex{
					Regex: rx,
					Score: p.Score,
				})
				pr.Validate = func(s string) bool {
					return s[0:2] != "00"
				}
				continue
			}

			// Skip any other look-around patterns
			if strings.Contains(rxText, "(?=") || strings.Contains(rxText, "(?!") || strings.Contains(rxText, "(?<") {
				fmt.Printf("⚠️ skipping unsupported lookaround in %s: %s\n", rec.Name, rxText)
				continue
			}

			// Compile normal regex
			rx, err := regexp.Compile(rxText)
			if err != nil {
				fmt.Printf("⚠️ skip invalid regex for %s: %v\n", rec.Name, err)
				continue
			}
			pr.Regexps = append(pr.Regexps, PatternRegex{
				Regex: rx,
				Score: p.Score,
			})
		}
		out = append(out, pr)
	}
	return out, nil
}

func isLuhnValid(d string) bool {
	sum, alt := 0, false
	for i := len(d) - 1; i >= 0; i-- {
		n := int(d[i] - '0')
		if alt {
			n *= 2
			if n > 9 {
				n -= 9
			}
		}
		sum += n
		alt = !alt
	}
	return sum%10 == 0
}

func (pr *PatternRecognizer) Recognize(text string, threshold float64) []RecognizerResult {
	var results []RecognizerResult

	// multiple regexps of same entity can give same matches for the text, so we need to deduplicate
	seen := make(map[string]struct{})

	for _, rx := range pr.Regexps {
		if rx.Score < threshold {
			continue
		}
		for _, loc := range rx.Regex.FindAllStringIndex(text, -1) {
			start, end := loc[0], loc[1]

			mapped, ok := entitiesMap[pr.EntityType]
			if !ok || mapped == "" {
				mapped = pr.EntityType
			}

			// Check if this combination has been seen
			matchKey := fmt.Sprintf("%s|%d|%d", mapped, start, end)
			if _, exists := seen[matchKey]; exists {
				continue
			}
			seen[matchKey] = struct{}{}

			match := text[start:end]

			if pr.EntityType == "CreditCardRecognizer" {
				digits := regexp.MustCompile(`\D`).ReplaceAllString(match, "")
				if !isLuhnValid(digits) {
					continue
				}
			}
			if pr.Validate != nil && !pr.Validate(match) {
				continue
			}
			results = append(results, RecognizerResult{
				EntityType: mapped,
				Match:      match,
				Score:      rx.Score,
				Start:      start,
				End:        end,
			})
		}
	}
	return results
}

type PresidioModel struct {
	recognizers []*PatternRecognizer
	threshold   float64
}

func NewPresidioModel() (*PresidioModel, error) {
	recs, err := loadPatterns()
	if err != nil {
		return nil, err
	}
	return &PresidioModel{
		recognizers: recs,
		threshold:   defaultPresidioThreshold,
	}, nil
}

func (m *PresidioModel) Predict(text string) ([]types.Entity, error) {
	var results []RecognizerResult
	for _, pr := range m.recognizers {
		results = append(results, pr.Recognize(text, m.threshold)...)
	}

	entities := make([]types.Entity, 0, len(results))
	for _, r := range results {
		entities = append(entities, types.CreateEntity(
			r.EntityType,
			text,
			r.Start,
			r.End,
		))
	}
	return entities, nil
}

func (m *PresidioModel) Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error {
	return fmt.Errorf("finetune not supported for presidio model")
}

func (m *PresidioModel) Save(path string, exportOnnx bool) error {
	return fmt.Errorf("save not supported for presidio model")
}

func (m *PresidioModel) Release() {}

func (m *PresidioModel) GetTags() []string {
	seen := make(map[string]struct{})
	tags := make([]string, 0, len(m.recognizers))
	for _, pr := range m.recognizers {
		mapped, ok := entitiesMap[pr.EntityType]
		if !ok || mapped == "" {
			mapped = pr.EntityType
		}

		if _, ok := seen[mapped]; !ok {
			tags = append(tags, mapped)
		}
	}

	return tags
}
