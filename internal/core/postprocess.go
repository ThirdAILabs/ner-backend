// core/postprocess.go
package core

import (
	"regexp"
	"strings"
	"unicode"
)

var (
	// exactly the Python PHONE_REGEX, with \d→[0-9], \s→space, no anchors
	phoneRegex = regexp.MustCompile(
		`(?:\+?[0-9]{1,3}[ .-]?)?` + // optional country code
			`(?:[0-9]{1,3}[ .-]?){2,5}` + // 2–5 groups of 1–3 digits + sep
			`[0-9]{1,4}` + // final block
			`(?: *(?:x|ext|extension) *[0-9]{1,6})?`, // optional extension
	)

	// exactly the Python CARD_REGEX
	cardRegex = regexp.MustCompile(
		`[0-9 \-]{12,19}`,
	)

	// exactly the Python CREDIT_SCORE_REGEX
	creditScoreRegex = regexp.MustCompile(
		`\b[0-9]{2,3}\b`,
	)

	// Python SSN_REGEX but without \1 (just allow any of the three separators)
	ssnRegex = regexp.MustCompile(
		`(?:[0-9]{3}[- .][0-9]{2}[- .][0-9]{4}|[0-9]{9})`,
	)
)

func FilterWordTags(text string, spans [][2]int, tags []string) []string {
	out := make([]string, len(tags))
	copy(out, tags)

	rules := []struct {
		label    string
		validate func(snippet, full string, s, e int) bool
		single   bool
	}{
		{"PHONENUMBER", func(sn, _ string, _, _ int) bool { return isValidPhone(sn) }, false},
		{"CARD_NUMBER", func(sn, _ string, _, _ int) bool { return isValidCard(sn) }, false},
		{"EMAIL", func(sn, _ string, _, _ int) bool { return isValidEmail(sn) }, true},
		{"SSN", func(sn, _ string, _, _ int) bool { return isValidSSN(sn) }, false},
		{"CREDIT_SCORE", isValidCreditScore, true},
	}

	for _, rule := range rules {
		for _, grp := range groupConsecutiveIndices(out, spans, rule.label, rule.single) {
			startIdx, endIdx := grp[0], grp[1]
			s, e := spans[startIdx][0], spans[endIdx][1]
			snippet := text[s:e]
			if !rule.validate(snippet, text, s, e) {
				for i := startIdx; i <= endIdx; i++ {
					out[i] = "O"
				}
			}
		}
	}

	return out
}

func isValidSSN(ssn string) bool {
	digits := stripNonDigits(ssn)
	if len(digits) != 9 {
		return false
	}
	return ssnRegex.MatchString(ssn)
}

func isValidPhone(num string) bool {
	digits := stripNonDigits(num)
	if len(digits) < 7 || len(digits) > 15 {
		return false
	}
	return phoneRegex.MatchString(num)
}

func isValidCard(num string) bool {
	digits := stripNonDigits(num)
	if len(digits) < 12 || len(digits) > 19 {
		return false
	}
	return luhnValid(digits)
}

func isValidCreditScore(score, full string, s, e int) bool {
	if !creditScoreRegex.MatchString(score) {
		return false
	}
	// context: 20 chars before and after
	startCtx := s - 20
	if startCtx < 0 {
		startCtx = 0
	}
	endCtx := e + 20
	if endCtx > len(full) {
		endCtx = len(full)
	}
	ctx := strings.ToLower(full[startCtx:s] + full[e:endCtx])
	return strings.Contains(ctx, "credit") && strings.Contains(ctx, "score")
}

func isValidEmail(email string) bool {
	parts := strings.SplitN(email, "@", 2)
	if len(parts) != 2 {
		return false
	}
	local, domain := parts[0], parts[1]
	if len(local) < 2 || len(domain) < 2 {
		return false
	}
	dom := strings.ToLower(domain)
	if dom == "localhost" {
		return true
	}
	return strings.Contains(domain, ".")
}

func stripNonDigits(s string) string {
	var b strings.Builder
	for _, r := range s {
		if unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func luhnValid(digits string) bool {
	sum := 0
	parity := len(digits) % 2
	for i, r := range digits {
		d := int(r - '0')
		if i%2 == parity {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		sum += d
	}
	return sum%10 == 0
}

func groupConsecutiveIndices(tags []string, spans [][2]int, label string, single bool) [][2]int {
	var groups [][2]int
	n := len(tags)
	for i := 0; i < n; {
		if tags[i] == label {
			start := i
			j := i
			for !single && j+1 < n &&
				tags[j+1] == label &&
				(spans[j+1][0] == spans[j][1] || spans[j+1][0] == spans[j][1]+1) {
				j++
			}
			groups = append(groups, [2]int{start, j})
			i = j + 1
		} else {
			i++
		}
	}
	return groups
}
