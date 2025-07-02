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
	if f, err := os.Create(o.trackUsageAt); err == nil {
		_ = json.NewEncoder(f).Encode(o.usage)
		f.Close()
	}

	// append full conversation to text log
	if f, err := os.OpenFile(o.responseFile, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err == nil {
		for _, m := range messages {
			f.WriteString("role: " + string(*m.GetRole()) + "\n")
			f.WriteString("content: " + m.OfFunction.Content.Value + "\n")
		}
		f.WriteString("Response: " + res.Choices[0].Message.Content + "\n")
		f.WriteString(fmt.Sprintf("Usage: %+v\n", tu))
		f.WriteString(strings.Repeat("=", 80) + "\n\n")
		f.Close()
	}
	o.mu.Unlock()

	return res.Choices[0].Message.Content, nil
}
