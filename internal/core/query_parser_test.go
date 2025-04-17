package core

import (
	"reflect"
	"testing"
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
	query := `label1 CONTAINS "value1" AND (label2 = "value2" OR NOT label3 > "value3")`
	expected := &AndFilter{
		filters: []Filter{
			&SubstringFilter{label: "label1", substr: "value1"},
			&OrFilter{
				filters: []Filter{
					&StringEqFilter{label: "label2", value: "value2"},
					&NotFilter{
						filter: &StringGtFilter{label: "label3", value: "value3"},
					},
				},
			},
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

func TestParseQuery_CountFilter(t *testing.T) {
	query := `COUNT label1 < 10`
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
