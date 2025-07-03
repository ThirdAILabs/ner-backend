package datagenv2

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	openai "github.com/openai/openai-go"
)

// TokenUsage tracks usage counts per model.
type TokenUsage struct {
	CompletionTokens int64 `json:"completion_tokens"`
	PromptTokens     int64 `json:"prompt_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// OpenAILLM wraps the OpenAI client, logs responses, and tracks usage.
type OpenAILLM struct {
	client       openai.Client
	model        string  // e.g. "gpt-4o"
	temp         float64 // temperature for generation
	trackUsageAt string
	responseFile string
	usage        map[string]*TokenUsage
	mu           sync.Mutex
}

// NewOpenAILLM creates a new wrapper.
// - apiKey: your OpenAI API key
// - baseURL: if non-empty, overrides the default API base URL
// - trackUsageAt: path to JSON file where token usage is written (defaults to "./usage.json" if empty)
func NewOpenAILLM(model, trackUsageAt string, temp float64) *OpenAILLM {
	// Default usage file
	if trackUsageAt == "" {
		trackUsageAt = filepath.Join(".", "usage.json")
	}
	// Log file for full responses
	respLog := filepath.Join(filepath.Dir(trackUsageAt), "response.txt")

	if err := os.MkdirAll(filepath.Dir(trackUsageAt), 0755); err != nil {
		slog.Warn("could not create usage dir", "dir", filepath.Dir(trackUsageAt), "error", err)
	}
	if err := os.MkdirAll(filepath.Dir(respLog), 0755); err != nil {
		slog.Warn("could not create response log dir", "dir", filepath.Dir(respLog), "error", err)
	}

	// Optionally, create empty files so you never hit “no such file” later
	if f, err := os.Create(trackUsageAt); err == nil {
		f.Close()
	}
	if f, err := os.Create(respLog); err == nil {
		f.Close()
	}

	return &OpenAILLM{
		client:       openai.NewClient(),
		trackUsageAt: trackUsageAt,
		responseFile: respLog,
		usage:        make(map[string]*TokenUsage),
		model:        model,
		temp:         temp,
	}
}

// Completion calls the chat completion endpoint, tracks usage, and logs everything.
// - model: e.g. "gpt-4o"
// - messages: built via openai.SystemMessage(...) and openai.UserMessage(...)
func (o *OpenAILLM) Generate(systemPrompt, prompt string, responseFormat openai.ChatCompletionNewParamsResponseFormatUnion) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Second)
	defer cancel()

	// build messages
	var messages []openai.ChatCompletionMessageParamUnion
	if systemPrompt != "" {
		messages = append(messages, openai.SystemMessage(systemPrompt))
	}
	messages = append(messages, openai.UserMessage(prompt))

	// call chat completion
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

	// Append full conversation to log
	if f, ferr := os.OpenFile(o.responseFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); ferr == nil {
		for _, m := range messages {
			role := string(*m.GetRole())
			if _, werr := f.WriteString("role: " + role + "\n"); werr != nil {
				slog.Warn("failed to write role", "error", werr)
			}
			content := m.OfFunction.Content.Value
			if _, werr := f.WriteString("content: " + content + "\n"); werr != nil {
				slog.Warn("failed to write content", "error", werr)
			}
		}
		// response
		resp := res.Choices[0].Message.Content
		if _, werr := f.WriteString("Response: " + resp + "\n"); werr != nil {
			slog.Warn("failed to write response", "error", werr)
		}
		// usage summary
		if _, werr := f.WriteString(fmt.Sprintf("Usage: %+v\n", tu)); werr != nil {
			slog.Warn("failed to write usage summary", "error", werr)
		}
		if _, werr := f.WriteString(strings.Repeat("=", 80) + "\n\n"); werr != nil {
			slog.Warn("failed to write separator", "error", werr)
		}
		f.Close()
	} else {
		slog.Warn("failed to open response log file", "error", ferr)
	}
	o.mu.Unlock()

	return res.Choices[0].Message.Content, nil
}
