package core

import (
	"testing"

	"ner-backend/internal/core/types"
)

func TestFilterEntities_PhoneNumbers(t *testing.T) {
	tests := []struct {
		name    string
		snippet string
		want    bool
	}{
		{"Valid 7-digit", "123-4567", true},
		{"Valid with country code", "+1 800 555 1234 ext 567", true},
		{"Too short", "12345", false},
		{"Too long", "12345678901234567890", false},
		{"Invalid letters", "ABC-DEF-GHIJ", false},
	}

	fullText := ""

	for _, tt := range tests {
		ent := types.Entity{
			Label: "PHONENUMBER",
			Text:  tt.snippet,
			Start: 0,
			End:   len(tt.snippet),
		}
		out := FilterEntities(fullText, []types.Entity{ent})
		got := len(out) == 1
		if got != tt.want {
			t.Errorf("Phone test %q: got %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestFilterEntities_CardNumbers(t *testing.T) {
	// Using Luhn‐valid numbers:
	//   - "4111 1111 1111 1111" is a common test Visa number.
	//   - "5500-0000-0000-0004" is a test Mastercard number.
	tests := []struct {
		name    string
		snippet string
		want    bool
	}{
		{"Valid Visa", "4111 1111 1111 1111", true},
		{"Valid Mastercard", "5500-0000-0000-0004", true},
		{"Too short", "1234 5678 901", false},
		{"Too long (20 digits)", "1234 5678 9012 3456 7890", false},
		{"Invalid Luhn", "4111 1111 1111 1112", false},
		{"Non‐digit chars", "abcd-efgh-ijkl-mnop", false},
	}

	fullText := ""

	for _, tt := range tests {
		ent := types.Entity{
			Label: "CARD_NUMBER",
			Text:  tt.snippet,
			Start: 0,
			End:   len(tt.snippet),
		}
		out := FilterEntities(fullText, []types.Entity{ent})
		got := len(out) == 1
		if got != tt.want {
			t.Errorf("Card test %q: got %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestFilterEntities_SSNs(t *testing.T) {
	tests := []struct {
		name    string
		snippet string
		want    bool
	}{
		{"Valid dashed", "123-45-6789", true},
		{"Valid spaced", "123 45 6789", true},
		{"Valid plain", "123456789", true},
		{"Too few digits", "123-45-678", false},
		{"Too many digits", "1234-56-7890", false},
		{"Wrong format", "12a-45-6789", false},
	}

	fullText := ""

	for _, tt := range tests {
		ent := types.Entity{
			Label: "SSN",
			Text:  tt.snippet,
			Start: 0,
			End:   len(tt.snippet),
		}
		out := FilterEntities(fullText, []types.Entity{ent})
		got := len(out) == 1
		if got != tt.want {
			t.Errorf("SSN test %q: got %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestFilterEntities_Emails(t *testing.T) {
	tests := []struct {
		name    string
		snippet string
		want    bool
	}{
		{"Valid simple", "john.doe@example.com", true},
		{"Valid localhost", "user@localhost", true},
		{"Missing @", "johndoeexample.com", false},
		{"Short local", "a@b.com", false}, // local part length < 2
		{"Short domain", "ab@c", false},   // domain length < 2 and no dot
		{"No dot in domain", "ab@domaincom", false},
	}

	fullText := ""

	for _, tt := range tests {
		ent := types.Entity{
			Label: "EMAIL",
			Text:  tt.snippet,
			Start: 0,
			End:   len(tt.snippet),
		}
		out := FilterEntities(fullText, []types.Entity{ent})
		got := len(out) == 1
		if got != tt.want {
			t.Errorf("Email test %q: got %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestFilterEntities_CreditScores(t *testing.T) {
	tests := []struct {
		name     string
		fullText string
		snippet  string
		start    int
		end      int
		want     bool
	}{
		{
			"Valid credit score – context present",
			"My credit score is 750 and rising.",
			"750",
			17, // index of '7' in the fullText string
			20,
			true,
		},
		{
			"No keyword 'credit'",
			"He scored 750 points yesterday.",
			"750",
			10,
			13,
			false,
		},
		{
			"No keyword 'score'",
			"This 800 number is high credit.",
			"800",
			5,
			8,
			false,
		},
		{
			"Too few digits",
			"My credit score is 5.",
			"5",
			17,
			18,
			false,
		},
		{
			"Too many digits",
			"Check credit score: 1000 exactly.",
			"1000",
			19,
			23,
			false,
		},
	}

	for _, tt := range tests {
		ent := types.Entity{
			Label: "CREDIT_SCORE",
			Text:  tt.snippet,
			Start: tt.start,
			End:   tt.end,
		}
		out := FilterEntities(tt.fullText, []types.Entity{ent})
		got := len(out) == 1
		if got != tt.want {
			t.Errorf("Credit‐Score test %q: got %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestFilterEntities_KeepOtherLabels(t *testing.T) {
	// If a label is not one of the five targeted ones, FilterEntities should preserve it unconditionally.
	ent := types.Entity{
		Label: "PERSON",
		Text:  "Alice",
		Start: 0,
		End:   5,
	}
	out := FilterEntities("Alice went home.", []types.Entity{ent})
	if len(out) != 1 {
		t.Errorf("Expected non‐target label to be preserved, but got %d entity(ies)", len(out))
		return
	}
	if out[0].Label != "PERSON" || out[0].Text != "Alice" {
		t.Errorf("Unexpected entity after filter: %+v", out[0])
	}
}
