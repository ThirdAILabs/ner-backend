package core

import (
	"ner-backend/internal/core/types"
	"regexp"
	"strings"
	"unicode"
)

var (
	phoneRegex = regexp.MustCompile(
		`(?:\+?[0-9]{1,3}[ .-]?)?` +
			`(?:[0-9]{1,3}[ .-]?){2,5}` +
			`[0-9]{1,4}` +
			`(?: *(?:x|ext|extension) *[0-9]{1,6})?`,
	)

	cardRegex = regexp.MustCompile(
		`[0-9 \-]{12,19}`,
	)

	creditScoreRegex = regexp.MustCompile(`\b[0-9]{2,3}\b`)

	ssnRegex = regexp.MustCompile(
		`(?:[0-9]{3}[- .][0-9]{2}[- .][0-9]{4}|[0-9]{9})`,
	)
)

func FilterEntities(fullText string, entities []types.Entity) []types.Entity {
	var out []types.Entity

	for _, ent := range entities {
		label := ent.Label
		snippet := ent.Text

		switch label {
		case "PHONENUMBER":
			if isValidPhone(snippet) {
				out = append(out, ent)
			}
		case "CARD_NUMBER":
			if isValidCard(snippet) {
				out = append(out, ent)
			}
		case "EMAIL":
			if isValidEmail(snippet) {
				out = append(out, ent)
			}
		case "SSN":
			if isValidSSN(snippet) {
				out = append(out, ent)
			}
		case "CREDIT_SCORE":
			if isValidCreditScore(snippet, fullText, ent.Start, ent.End) {
				out = append(out, ent)
			}
		default:
			out = append(out, ent)
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
