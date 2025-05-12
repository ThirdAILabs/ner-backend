package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"time"

	"ner-backend/internal/core"
	"ner-backend/pkg/api"
)

var sentence = "John lives in London."

func run() error {
	os.Setenv("PYTHON_EXECUTABLE_PATH", "/opt/conda/envs/pii-presidio-3.10/bin/python3")
	os.Setenv("PYTHON_MODEL_PLUGIN_SCRIPT_PATH", "/home/ubuntu/shubh/ner/ner-backend/plugin/plugin-python/plugin.py")
	model, err := core.LoadModel("transformer", "/home/ubuntu/shubh/ner/misc/ner-models/transformer_model")
	if err != nil {
		return fmt.Errorf("error loading model: %w", err)
	}
	defer model.Release()

	fmt.Println("→ Initial prediction")
	initial, err := model.Predict(sentence)
	if err != nil {
		return fmt.Errorf("predict: %w", err)
	}
	fmt.Printf("  Entities: %+v\n\n", initial)

	tags := []api.TagInfo{
		{
			Name:        "ADDRESS",
			Description: "Place names",
			Examples:    []string{"London", "New York"},
		},
	}
	samples := []api.Sample{
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
		{
			Tokens: []string{"John", "lives", "in", "London."},
			Labels: []string{"NAME", "O", "O", "ADDRESS"},
		},
	}

	// 4) Finetune
	fmt.Println("→ Finetuning...")
	start := time.Now()
	if err := model.Finetune("Add LOCATION tag", tags, samples); err != nil {
		return fmt.Errorf("finetune: %w", err)
	}
	fmt.Printf("  Done in %v\n\n", time.Since(start))

	// 6) Post‐finetune prediction
	fmt.Println("→ Post-finetune prediction")
	after, err := model.Predict(sentence)
	if err != nil {
		return fmt.Errorf("predict after finetune: %w", err)
	}
	fmt.Printf("  Entities: %+v\n", after)

	return nil
}

func main() {
	// Silence plugin logs
	log.SetOutput(ioutil.Discard)

	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %+v\n", err)
		os.Exit(1)
	}
}
