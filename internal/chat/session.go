package chat

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"gorm.io/gorm"
)

type ChatSession struct {
	mu           sync.Mutex
	db           *gorm.DB
	sessionID    string
	model        string
	apiKey       string
	openAIClient *openai.LLM
}

func NewChatSession(db *gorm.DB, sessionID, model, apiKey string) (*ChatSession, error) {
	client, err := openai.New(openai.WithToken(apiKey), openai.WithModel(model))
	if err != nil {
		return nil, fmt.Errorf("could not create OpenAI client: %v", err)
	}

	return &ChatSession{
		db:           db,
		sessionID:    sessionID,
		model:        model,
		apiKey:       apiKey,
		openAIClient: client,
	}, nil
}

func (session *ChatSession) Chat(userInput string) (string, error) {
	session.mu.Lock()
	defer session.mu.Unlock()

	if err := session.saveMessage("user", userInput); err != nil {
		return "", err
	}

	history, err := session.getChatHistory()
	if err != nil {
		return "", err
	}

	context := ""
	for _, msg := range history {
		context += fmt.Sprintf("%s: %s\n", msg.MessageType, msg.Content)
	}

	openAIResponse, err := session.getOpenAIResponse(context + "user: " + userInput)
	if err != nil {
		return "", err
	}

	if err := session.saveMessage("ai", openAIResponse); err != nil {
		return "", err
	}

	return openAIResponse, nil
}

func (session *ChatSession) getOpenAIResponse(ctx string) (string, error) {
	messages := []llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, "You are a helpful assistant."),
		llms.TextParts(llms.ChatMessageTypeHuman, ctx),
	}

	resp, err := session.openAIClient.GenerateContent(context.Background(), messages)
	if err != nil {
		log.Printf("Error calling OpenAI API: %v", err)
		return "", err
	}

	return resp.Choices[0].Content, nil
}

func (session *ChatSession) saveMessage(messageType, content string) error {
	chatMessage := ChatHistory{
		SessionID:   session.sessionID,
		MessageType: messageType,
		Content:     content,
	}
	return session.db.Create(&chatMessage).Error
}

func (session *ChatSession) getChatHistory() ([]ChatHistory, error) {
	var history []ChatHistory
	err := session.db.Where("session_id = ?", session.sessionID).Order("timestamp ASC").Find(&history).Error
	if err != nil {
		return nil, err
	}
	return history, nil
}
