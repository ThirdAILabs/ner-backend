package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"log/slog"
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

type TagMetadata struct {
	TagMap map[string]string 
	Assigned map[string]string
	LabelCounts map[string]int
}

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

func (session *ChatSession) Redact(text string, tagMetadata TagMetadata) (string, TagMetadata, error) {
	entities, err := session.ner.Predict(text)
	if err != nil {
		return "", TagMetadata{}, fmt.Errorf("error predicting entities: %w", err)
	}

	sort.Slice(entities, func(i, j int) bool {
		if entities[i].Start == entities[j].Start {
			return entities[i].End > entities[j].End
		}
		return entities[i].Start < entities[j].Start
	})

	var b strings.Builder
	cursor := 0

	for _, ent := range entities {

		if ent.Start < cursor || ent.End > len(text) {
			continue
		}

		b.WriteString(text[cursor:ent.Start])

		key := fmt.Sprintf("%s_%s", ent.Text, ent.Label)
		userTag, ok := tagMetadata.Assigned[key]
		if !ok {
			tagMetadata.LabelCounts[ent.Label]++
			userTag = fmt.Sprintf("[%s_%d]", ent.Label, tagMetadata.LabelCounts[ent.Label])
			tagMetadata.Assigned[key] = userTag
			tagMetadata.TagMap[userTag] = ent.Text
		}

		b.WriteString(userTag)
		cursor = ent.End
	}

	b.WriteString(text[cursor:])
	return b.String(), tagMetadata, nil
}

type ChatIterator func(yield func(string, string, map[string]string, error) bool)

func (session *ChatSession) ChatStream(userInput string) ChatIterator {
	return func(yield func(string, string, map[string]string, error) bool) {
		slog.Info("1")
		session.mu.Lock()
		defer session.mu.Unlock()

		slog.Info("2")
		tagMetadata, err := session.getTagMetadata()
		if err != nil {
			yield("", "", nil, fmt.Errorf("error getting tag metadata: %v", err))
			return
		}

		slog.Info("3")
		redactedText, newTagMetadata, err := session.Redact(userInput, tagMetadata)
		if err != nil {
			yield("", "", nil, fmt.Errorf("error redacting user input: %v", err))
			return
		}

		slog.Info("4")
		if err := session.updateTagMetadata(newTagMetadata); err != nil {
			yield("", "", nil, fmt.Errorf("error updating tag map: %v", err))
			return
		}

		slog.Info("5")
		// First yield the non-streaming components
		// TODO: Will this the tag map get too big? Should we only yield
		// the subset of the tag map that is relevant to the current message?
		if !yield(redactedText, "", newTagMetadata.TagMap, nil) {
			log.Printf("Failed to yield redacted text and tag map")
			return
		}

		slog.Info("6")
		history, err := session.getChatHistory()
		if err != nil {
			yield("", "", nil, err)
			return
		}
		
		slog.Info("7")
		context := ""
		for _, msg := range history {
			context += fmt.Sprintf("%s: %s\n", msg.MessageType, msg.Content)
		}
		context += fmt.Sprintf("User: %s\n", redactedText)

		openaiResp := ""

		slog.Info("8")
		// Then stream the OpenAI response
		session.streamOpenAIResponse(context)(func (chunk string, err error) bool {
			if err != nil {
				return yield("", "", nil, err)
			}
			openaiResp += chunk
			return yield("", chunk, nil, nil)
		})
		
		slog.Info("9")
		// Only save messages if the whole process was successful.
		// This gives the illusion of atomicity; a request either succeeds or fails entirely.
		// We still yield the error so the frontend can process accordingly.
		if err := session.saveMessage("user", redactedText, nil); err != nil {
			yield("", "", nil, err)
			return
		}
		
		slog.Info("10")
		if err := session.saveMessage("ai", openaiResp, nil); err != nil {
			yield("", "", nil, err)
			return
		}
	}
}

func (session *ChatSession) getTagMetadata() (TagMetadata, error) {
	slog.Info("A")
	var chatSession database.ChatSession
	err := session.db.Where("id = ?", session.sessionID).First(&chatSession).Error
	if err != nil {
		return TagMetadata{}, err
	}
	
	slog.Info("B")
	if chatSession.TagMetadata == nil {
		return TagMetadata{
			TagMap: make(map[string]string),
			Assigned: make(map[string]string),
			LabelCounts: make(map[string]int),
		}, nil
	}
		
	slog.Info("C")
	var tagMetadata TagMetadata
	if err := json.Unmarshal(chatSession.TagMetadata, &tagMetadata); err != nil {
		return TagMetadata{}, err
	}
	
	slog.Info("D")
	return tagMetadata, nil
}

func (session *ChatSession) updateTagMetadata(tagMetadata TagMetadata) error {
	tagMetadataJSON, err := json.Marshal(tagMetadata)
	slog.Info("UpdateTagMetadata", "tagMetadata", tagMetadataJSON);
	if err != nil {
		return fmt.Errorf("error marshalling tag map: %v", err)
	}
	return session.db.Model(&database.ChatSession{}).Where("id = ?", session.sessionID).Update("tag_metadata", tagMetadataJSON).Error
}

func (session *ChatSession) streamOpenAIResponse(ctx string) func(yield func(string, error) bool) {
	return func(yield func(string, error) bool) {
		messages := []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, "You are a helpful assistant. Note that some fields will be obfuscated (e.g. [PERSON_1]) and will be injected back when it is displayed to the reader, so pretend like you know what they are."),
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
