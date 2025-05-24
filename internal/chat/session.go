package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"sort"
	"strings"
	"sync"

	"github.com/google/uuid"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"gorm.io/datatypes"
	"gorm.io/gorm"
)

type ChatSession struct {
	mu           sync.Mutex
	db           *gorm.DB
	sessionID    uuid.UUID
	model        string
	apiKey       string
	openAIClient *openai.LLM
	ner          core.Model
}

func NewChatSession(db *gorm.DB, sessionID uuid.UUID, model, apiKey string, ner core.Model) (*ChatSession, error) {
	var sessions []database.ChatSession
	err := db.Where("id = ?", sessionID).Find(&sessions).Error
	if err != nil {
		return nil, err
	}
	if len(sessions) == 0 {
		return nil, fmt.Errorf("session not found")
	}

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
		return "", nil, fmt.Errorf("error predicting entities: %w", err)
	}

	sort.Slice(entities, func(i, j int) bool {
		if entities[i].Start == entities[j].Start {
			return entities[i].End > entities[j].End
		}
		return entities[i].Start < entities[j].Start
	})

	var b strings.Builder
	cursor := 0
	tagMap := make(map[string]string)

	labelCounts := make(map[string]int)

	assigned := make(map[string]string)

	for _, ent := range entities {

		if ent.Start < cursor || ent.End > len(text) {
			continue
		}

		b.WriteString(text[cursor:ent.Start])

		key := fmt.Sprintf("%s_%s", ent.Text, ent.Label)
		userTag, ok := assigned[key]
		if !ok {
			labelCounts[ent.Label]++
			userTag = fmt.Sprintf("[%s_%d]", ent.Label, labelCounts[ent.Label])
			assigned[key] = userTag
			tagMap[userTag] = ent.Text
		}

		b.WriteString(userTag)
		cursor = ent.End
	}

	b.WriteString(text[cursor:])
	return b.String(), tagMap, nil
}

func (session *ChatSession) Chat(userInput string) (string, string, map[string]string, error) {
	session.mu.Lock()
	defer session.mu.Unlock()

	redactedText, tagMap, err := session.Redact(userInput)
	if err != nil {
		return "", "", nil, fmt.Errorf("error redacting user input: %v", err)
	}
	
	history, err := session.getChatHistory()
	if err != nil {
		return "", "", nil, err
	}
	
	context := ""
	for _, msg := range history {
		context += fmt.Sprintf("%s: %s\n", msg.MessageType, msg.Content)
	}
	context += fmt.Sprintf("User: %s\n", redactedText)
	
	openaiResp, err := session.getOpenAIResponse(context)
	if err != nil {
		return "", "", nil, err
	}
	
	// Only save messages if the whole process was successful.
	// This gives the illusion of atomicity; a request either succeeds or fails entirely.
	if err := session.saveMessage("user", redactedText, tagMap); err != nil {
		return "", "", nil, err
	}

	if err := session.saveMessage("ai", openaiResp, nil); err != nil {
		return "", "", nil, err
	}

	return redactedText, openaiResp, tagMap, nil
}

type ChatIterator func(yield func(string, string, map[string]string, error) bool)

func (session *ChatSession) ChatStream(userInput string) ChatIterator {
	return func(yield func(string, string, map[string]string, error) bool) {
		session.mu.Lock()
		defer session.mu.Unlock()
	
		redactedText, tagMap, err := session.Redact(userInput)
		if err != nil {
			yield("", "", nil, fmt.Errorf("error redacting user input: %v", err))
			return
		}
		
		// First yield the non-streaming components
		if !yield(redactedText, "", tagMap, nil) {
			log.Printf("Failed to yield redacted text and tag map")
			return
		}
		
		history, err := session.getChatHistory()
		if err != nil {
			yield("", "", nil, err)
			return
		}
		
		context := ""
		for _, msg := range history {
			context += fmt.Sprintf("%s: %s\n", msg.MessageType, msg.Content)
		}
		context += fmt.Sprintf("User: %s\n", redactedText)

		// Then stream the OpenAI response
		session.streamOpenAIResponse(context)(func (chunk string, err error) bool {
			return yield("", chunk, nil, nil)
		})
		openaiResp, err := session.getOpenAIResponse(context)
		if err != nil {
			yield("", "", nil, err)
			return
		}
		
		// Only save messages if the whole process was successful.
		// This gives the illusion of atomicity; a request either succeeds or fails entirely.
		// We still yield the error so the frontend can process accordingly.
		if err := session.saveMessage("user", redactedText, tagMap); err != nil {
			yield("", "", nil, err)
			return
		}
	
		if err := session.saveMessage("ai", openaiResp, nil); err != nil {
			yield("", "", nil, err)
			return
		}
	}
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

func (session *ChatSession) streamOpenAIResponse(ctx string) func(yield func(string, error) bool) {
	return func(yield func(string, error) bool) {
		messages := []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, "You are a helpful assistant."),
			llms.TextParts(llms.ChatMessageTypeHuman, ctx),
		}

		var yieldSuccess bool
		
		_, err := session.openAIClient.GenerateContent(context.Background(), messages, llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			if yieldSuccess = yield(string(chunk), nil); !yieldSuccess {
				return fmt.Errorf("failed to yield chunk")
			}
			return nil
		}))

		if !yieldSuccess {
			log.Printf("Failed to yield chunk")
			return
		}

		if err != nil {
			log.Printf("Error calling OpenAI API: %v", err)
			yieldSuccess = yield("", err)
		}
	}
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
