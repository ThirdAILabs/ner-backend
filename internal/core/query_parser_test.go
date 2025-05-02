package core

import (
	"math"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseQuery_SimpleFilter(t *testing.T) {
	query := `label1 CONTAINS "value"`
	expected := &SubstringFilter{label: "label1", substr: "value"}

	filter, err := ParseQuery(query)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(filter, expected) {
		t.Errorf("expected %v, got %v", expected, filter)
	}
}

func TestParseQuery_AndExpression(t *testing.T) {
	query := `label1 CONTAINS "value1" AND label2 = "value2"`
	expected := &AndFilter{
		filters: []Filter{
			&SubstringFilter{label: "label1", substr: "value1"},
			&StringEqFilter{label: "label2", value: "value2"},
		},
	}

	filter, err := ParseQuery(query)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(filter, expected) {
		t.Errorf("expected %v, got %v", expected, filter)
	}
}

func TestParseQuery_OrExpression(t *testing.T) {
	query := `label1 CONTAINS "value1" OR label2 = "value2"`
	expected := &OrFilter{
		filters: []Filter{
			&SubstringFilter{label: "label1", substr: "value1"},
			&StringEqFilter{label: "label2", value: "value2"},
		},
	}

	filter, err := ParseQuery(query)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(filter, expected) {
		t.Errorf("expected %v, got %v", expected, filter)
	}
}

func TestParseQuery_NotExpression(t *testing.T) {
	query := `NOT label1 CONTAINS "value"`
	expected := &NotFilter{
		filter: &SubstringFilter{label: "label1", substr: "value"},
	}

	filter, err := ParseQuery(query)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(filter, expected) {
		t.Errorf("expected %v, got %v", expected, filter)
	}
}

func TestParseQuery_ComplexExpression(t *testing.T) {
	query := `label1 CONTAINS "value1" AND (label2 = "value2" OR NOT COUNT(label3) > 4)`
	expected := &AndFilter{
		filters: []Filter{
			&SubstringFilter{label: "label1", substr: "value1"},
			&OrFilter{
				filters: []Filter{
					&StringEqFilter{label: "label2", value: "value2"},
					&NotFilter{
						filter: &CountFilter{label: "label3", min: 4, max: math.MaxInt},
					},
				},
			},
		},
	}

	filter, err := ParseQuery(query)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(t, filter, expected)

	if !reflect.DeepEqual(filter, expected) {
		t.Errorf("expected %v, got %v", expected, filter)
	}
}

func TestParseQuery_CountFilter(t *testing.T) {
	query := `COUNT(label1) < 10`
	expected := &CountFilter{label: "label1", min: -1, max: 10}

	filter, err := ParseQuery(query)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(filter, expected) {
		t.Errorf("expected %v, got %v", expected, filter)
	}
}

func TestParseQuery_InvalidQuery(t *testing.T) {
	query := `label1 CONTAINS`
	_, err := ParseQuery(query)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
}
