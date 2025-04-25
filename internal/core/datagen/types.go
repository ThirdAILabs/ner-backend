package datagen

import (
	"fmt"
	"strings"
)

type Sample struct {
	Tokens []string
	Labels []string
}

func (s *Sample) asTemplate() string {
	output := make([]string, 0, len(s.Tokens))
	for i, token := range s.Tokens {
		if s.Labels[i] == "O" {
			output = append(output, token)
		} else {
			output = append(output, fmt.Sprintf("[%s]", s.Labels[i]))
		}
	}
	return strings.Join(output, " ")
}

type TagInfo struct {
	Name        string
	Description string
	Examples    []string
}

type tagInfoExtendedDescription struct {
	Name                string
	Description         string
	ExtendedDescription string
	Examples            []string
}
