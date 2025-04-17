package core

import (
	"strings"
)

type LabelToEntities map[string][]Entity

type Filter interface {
	Matches(entities LabelToEntities) bool
}

type AndFilter struct {
	filters []Filter
}

func (f *AndFilter) Matches(entities LabelToEntities) bool {
	for _, filter := range f.filters {
		if !filter.Matches(entities) {
			return false
		}
	}
	return true
}

type OrFilter struct {
	filters []Filter
}

func (f *OrFilter) Matches(entities LabelToEntities) bool {
	for _, filter := range f.filters {
		if filter.Matches(entities) {
			return true
		}
	}
	return false
}

type NotFilter struct {
	filter Filter
}

func (f *NotFilter) Matches(entities LabelToEntities) bool {
	return !f.filter.Matches(entities)
}

type CountFilter struct {
	label string
	min   int
	max   int
}

func (f *CountFilter) Matches(entities LabelToEntities) bool {
	count := len(entities[f.label])
	return f.min < count && count < f.max
}

type SubstringFilter struct {
	label  string
	substr string
}

func (f *SubstringFilter) Matches(entities LabelToEntities) bool {
	for _, entity := range entities[f.label] {
		if strings.Contains(entity.Text, f.substr) {
			return true
		}
	}
	return false
}

type StringEqFilter struct {
	label string
	value string
}

func (f *StringEqFilter) Matches(entities LabelToEntities) bool {
	for _, entity := range entities[f.label] {
		if entity.Text == f.value {
			return true
		}
	}
	return false
}

type StringLtFilter struct {
	label string
	value string
}

func (f *StringLtFilter) Matches(entities LabelToEntities) bool {
	for _, entity := range entities[f.label] {
		if entity.Text < f.value {
			return true
		}
	}
	return false
}

type StringGtFilter struct {
	label string
	value string
}

func (f *StringGtFilter) Matches(entities LabelToEntities) bool {
	for _, entity := range entities[f.label] {
		if entity.Text > f.value {
			return true
		}
	}
	return false
}
