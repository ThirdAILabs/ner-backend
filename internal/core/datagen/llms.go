package datagen

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/openai/openai-go"
)

type LLM interface {
	Generate(systemPrompt, userPrompt string) (string, error)
}

type OpenAI struct {
	client openai.Client
	model  string
	temp   float64
}

func NewOpenAI(model string, temp float64) *OpenAI {
	return &OpenAI{
		client: openai.NewClient(),
		model:  model,
		temp:   temp,
	}
}

func (o *OpenAI) Generate(systemPrompt, prompt string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Second)
	defer cancel()

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, 2)

	if len(systemPrompt) > 0 {
		messages = append(messages, openai.SystemMessage(systemPrompt))
	}
	messages = append(messages, openai.UserMessage(prompt))

	chatOpts := openai.ChatCompletionNewParams{
		Messages:    messages,
		Model:       o.model,
		Temperature: openai.Float(o.temp),
	}

	res, err := o.client.Chat.Completions.New(ctx, chatOpts)
	if err != nil {
		slog.Error("openai error: chat completions failed", "error", err)
		return "", fmt.Errorf("openai generation failed: %w", err)
	}

	return res.Choices[0].Message.Content, nil
}
