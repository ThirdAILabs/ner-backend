package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"sort"
	"sync"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type ChatSession struct {
	mu           sync.Mutex
	db           *gorm.DB
	sessionID    string
	model        string
	apiKey       string
	openAIClient *openai.LLM
	ner          core.Model
}

func NewChatSession(db *gorm.DB, sessionID, model, apiKey string, ner core.Model) (*ChatSession, error) {
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
		ner:          ner,
	}, nil
}

func (session *ChatSession) Redact(text string) (string, map[string]string, error) {
	entities, err := session.ner.Predict(text)
	if err != nil {
		return "", nil, fmt.Errorf("error predicting entities: %v", err)
	}

	redactedText := text
	// user-tag --> entity-text
	userTagMap := make(map[string]string)

	// map to store unique tags for each entity label
	uniqueTagId := make(map[string]int)

	// map to store if (entity, label) is seen and what is the user-tag mapped for it.
	// This is being used because same entity text can be tagged as different label in different parts of the text.
	tempMp := make(map[string]string)

	sort.Slice(entities, func(i, j int) bool {
		return entities[i].Start > entities[j].Start
	})

	for _, entity := range entities {
		var userTag string
		key := fmt.Sprintf("%s_%s", entity.Text, entity.Label)
		if existingTag, exists := tempMp[key]; exists {
			userTag = existingTag
		} else {
			uniqueTagId[entity.Label]++
			userTag = fmt.Sprintf("[%s_%d]", entity.Label, uniqueTagId[key])
			tempMp[key] = userTag
			userTagMap[userTag] = entity.Text
		}

		redactedText = redactedText[:entity.Start] + userTag + redactedText[entity.End:]
	}
	return redactedText, userTagMap, nil
}

func (session *ChatSession) Chat(userInput string) (string, map[string]string, error) {
	session.mu.Lock()
	defer session.mu.Unlock()

	redactedText, tagMap, err := session.Redact(userInput)
	if err != nil {
		return "", nil, fmt.Errorf("error redacting user input: %v", err)
	}

	if err := session.saveMessage("user", redactedText, tagMap); err != nil {
		return "", nil, err
	}

	history, err := session.getChatHistory()
	if err != nil {
		return "", nil, err
	}

	context := ""
	for _, msg := range history {
		context += fmt.Sprintf("%s: %s\n", msg.MessageType, msg.Content)
	}

	openaiResp, err := session.getOpenAIResponse(context)
	if err != nil {
		return "", nil, err
	}

	if err := session.saveMessage("ai", openaiResp, nil); err != nil {
		return "", nil, err
	}

	return openaiResp, tagMap, nil
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

func (session *ChatSession) saveMessage(messageType, content string, metadata map[string]string) error {
	var metadataJSON datatypes.JSON = nil
	if metadata != nil {
		b, err := json.Marshal(metadata)
		if err != nil {
			return fmt.Errorf("could not marshal metadata: %v", err)
		}
		metadataJSON = datatypes.JSON(b)
	}

	chatMessage := database.ChatHistory{
		SessionID:   session.sessionID,
		MessageType: messageType,
		Content:     content,
		Metadata:    metadataJSON,
	}
	return session.db.Create(&chatMessage).Error
}

func (session *ChatSession) getChatHistory() ([]database.ChatHistory, error) {
	var history []database.ChatHistory
	err := session.db.Where("session_id = ?", session.sessionID).Order("timestamp ASC").Find(&history).Error
	if err != nil {
		return nil, err
	}
	return history, nil
}
