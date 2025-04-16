package core

import "context"

type Entity struct {
	Label string
	Text  string
	Start int // TODO how should this be represented?
	End   int
}

type Model interface {
	Predict(ctx context.Context, text string) ([]Entity, error)
}
