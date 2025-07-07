package datagenv2

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	openai "github.com/openai/openai-go"
)

type TokenUsage struct {
	CompletionTokens int64 `json:"completion_tokens"`
	PromptTokens     int64 `json:"prompt_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type OpenAILLM struct {
	client       openai.Client
	model        string  // e.g. "gpt-4o"
	temp         float64 // temperature for generation
	trackUsageAt string
	usage        map[string]*TokenUsage
	mu           sync.Mutex
}

func NewOpenAILLM(model, trackUsageAt string, temp float64) *OpenAILLM {
	// Default usage file
	if trackUsageAt == "" {
		trackUsageAt = filepath.Join(".", "usage.json")
	}

	if err := os.MkdirAll(filepath.Dir(trackUsageAt), 0755); err != nil {
		slog.Warn("could not create usage dir", "dir", filepath.Dir(trackUsageAt), "error", err)
	}

	// Optionally, create empty files so you never hit “no such file” later
	if f, err := os.Create(trackUsageAt); err == nil {
		f.Close()
	}

	return &OpenAILLM{
		client:       openai.NewClient(),
		trackUsageAt: trackUsageAt,
		usage:        make(map[string]*TokenUsage),
		model:        model,
		temp:         temp,
	}
}

func (o *OpenAILLM) Generate(systemPrompt, prompt string, responseFormat openai.ChatCompletionNewParamsResponseFormatUnion) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Second)
	defer cancel()

	var messages []openai.ChatCompletionMessageParamUnion
	if systemPrompt != "" {
		messages = append(messages, openai.SystemMessage(systemPrompt))
	}
	messages = append(messages, openai.UserMessage(prompt))

	chatReq := openai.ChatCompletionNewParams{
		Model:       o.model,
		Messages:    messages,
		Temperature: openai.Float(o.temp),
	}
	chatReq.ResponseFormat = responseFormat

	res, err := o.client.Chat.Completions.New(ctx, chatReq)
	if err != nil {
		slog.Error("openai error: chat completions failed", "error", err)
		return "", fmt.Errorf("openai generation failed: %w", err)
	}

	// update usage
	o.mu.Lock()
	tu, ok := o.usage[o.model]
	if !ok {
		tu = &TokenUsage{}
		o.usage[o.model] = tu
	}
	tu.CompletionTokens += res.Usage.CompletionTokens
	tu.PromptTokens += res.Usage.PromptTokens
	tu.TotalTokens += res.Usage.TotalTokens

	// write usage JSON
	if f, ferr := os.Create(o.trackUsageAt); ferr == nil {
		if jerr := json.NewEncoder(f).Encode(o.usage); jerr != nil {
			slog.Warn("failed to write usage JSON", "error", jerr)
		}
		f.Close()
	} else {
		slog.Warn("failed to create usage file", "error", ferr)
	}

	o.mu.Unlock()

	return res.Choices[0].Message.Content, nil
}
