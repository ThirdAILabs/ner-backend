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
	LContext   string
	RContext   string
}

type PatternRecognizer struct {
	EntityType string
	Regexps    []*regexp.Regexp
	Score      float64
	Validate   func(string) bool
}

var entitiesMap = map[string]string{
	"LOCATION":                        "ADDRESS",
	"UsLicenseRecognizer":             "VIN", // AKA US_DRIVER_LICENSE
	"PHONE_NUMBER":                    "PHONENUMBER",
	"DATE_TIME":                       "DATE",
	"EMAIL_ADDRESS":                   "EMAIL",
	"CreditCardRecognizer":            "CARD_NUMBER",
	"UsSsnRecognizer":                 "SSN",
	"URL":                             "URL",
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
			Score:      0,
			Validate:   nil,
		}
		for _, p := range rec.Patterns {
			rxText := p.Regex

			// PAN (InPanRecognizer) low-strength lookahead
			if rec.Name == "InPanRecognizer" && strings.Contains(rxText, "(?=") {
				base := `\b[\w@#$%^?~-]{10}\b`
				rx := regexp.MustCompile(base)
				pr.Regexps = append(pr.Regexps, rx)
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
				if p.Score > pr.Score {
					pr.Score = p.Score
				}
				continue
			}

			// Vehicle reg part 1: (I)(?!00000)\d{5}
			if rec.Name == "InVehicleRegistrationRecognizer" && strings.Contains(rxText, "(?!00000)") {
				base := `\bI[0-9]{5}\b`
				rx := regexp.MustCompile(base)
				pr.Regexps = append(pr.Regexps, rx)
				pr.Validate = func(s string) bool {
					return s[1:] != "00000"
				}
				if p.Score > pr.Score {
					pr.Score = p.Score
				}
				continue
			}

			// Vehicle reg part 2: (?!00)\d{2}[A-FH-KPRX]\d{6}[A-Z]
			if rec.Name == "InVehicleRegistrationRecognizer" && strings.Contains(rxText, "(?!00)") {
				base := `\b[0-9]{2}[A-FH-KPRX][0-9]{6}[A-Z]\b`
				rx := regexp.MustCompile(base)
				pr.Regexps = append(pr.Regexps, rx)
				pr.Validate = func(s string) bool {
					return s[0:2] != "00"
				}
				if p.Score > pr.Score {
					pr.Score = p.Score
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
			pr.Regexps = append(pr.Regexps, rx)
			if p.Score > pr.Score {
				pr.Score = p.Score
			}
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
	const ctxLen = 20
	var results []RecognizerResult
	for _, rx := range pr.Regexps {
		for _, loc := range rx.FindAllStringIndex(text, -1) {
			start, end := loc[0], loc[1]
			match := text[start:end]
			score := pr.Score

			if pr.EntityType == "CreditCardRecognizer" {
				digits := regexp.MustCompile(`\D`).ReplaceAllString(match, "")
				if !isLuhnValid(digits) {
					continue
				}
			}
			if score < threshold {
				continue
			}
			if pr.Validate != nil && !pr.Validate(match) {
				continue
			}

			lctxStart := start - ctxLen
			if lctxStart < 0 {
				lctxStart = 0
			}
			rctxEnd := end + ctxLen
			if rctxEnd > len(text) {
				rctxEnd = len(text)
			}
			lctx := text[lctxStart:start]
			rctx := text[end:rctxEnd]

			mapped, ok := entitiesMap[pr.EntityType]
			if !ok || mapped == "" {
				mapped = pr.EntityType
			}
			results = append(results, RecognizerResult{
				EntityType: mapped,
				Match:      match,
				Score:      score,
				Start:      loc[0],
				End:        loc[1],
				LContext:   lctx,
				RContext:   rctx,
			})
		}
	}
	return results
}

func analyze(text string, threshold float64) []RecognizerResult {
	recs, err := loadPatterns()
	if err != nil {
		fmt.Printf("⚠️ failed to load recognizers: %v\n", err)
		return nil
	}
	var out []RecognizerResult
	for _, pr := range recs {
		out = append(out, pr.Recognize(text, threshold)...)
	}
	return out
}

type presidioModel struct {
	threshold float64
}

func (m *presidioModel) Predict(text string) ([]types.Entity, error) {
	results := analyze(text, m.threshold)
	out := make([]types.Entity, 0, len(results))
	for _, r := range results {
		out = append(out, types.Entity{
			Text:     r.Match,
			Label:    r.EntityType,
			Start:    r.Start,
			End:      r.End,
			LContext: r.LContext,
			RContext: r.RContext,
		})
	}
	return out, nil
}

func (m *presidioModel) Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error {
	return fmt.Errorf("finetune not supported for presidio model")
}

func (m *presidioModel) Save(path string) error {
	return fmt.Errorf("save not supported for presidio model")
}

func (m *presidioModel) Release() {}
