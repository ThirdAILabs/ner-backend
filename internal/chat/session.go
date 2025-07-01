package chat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"ner-backend/internal/core"
	"ner-backend/internal/database"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"gorm.io/datatypes"
)

var ErrStopStream = errors.New("stop stream")

type TagMetadata struct {
	TagMap      map[string]string
	Assigned    map[string]string
	LabelCounts map[string]int
}

func NewTagMetadata() TagMetadata {
	return TagMetadata{
		TagMap:      make(map[string]string),
		Assigned:    make(map[string]string),
		LabelCounts: make(map[string]int),
	}
}

type ChatSession struct {
	db           *ChatDB
	sessionID    uuid.UUID
	model        string
	apiKey       string
	openAIClient *openai.LLM
	ner          core.Model
}

func NewChatSession(db *ChatDB, sessionID uuid.UUID, model, apiKey string, ner core.Model) (*ChatSession, error) {
	_, err := db.GetSession(sessionID)
	if err != nil {
		return nil, err
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

func NewExtensionChatSession(db *ChatDB, sessionID uuid.UUID, ner core.Model) ChatSession {
	return ChatSession{
		db:           db,
		sessionID:    sessionID,
		ner:          ner,
	}
}

func (session *ChatSession) redact(text string, tagMetadata TagMetadata) (string, TagMetadata, error) {
	entities, err := session.ner.Predict(text)
	if err != nil {
		return "", TagMetadata{}, fmt.Errorf("error predicting entities: %w", err)
	}
	entities = core.FilterEntities(text, entities)

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

func (session *ChatSession) Redact(text string) (string, error) {
	tagMetadata, err := session.getTagMetadata()
	if err != nil {
		return "", fmt.Errorf("error getting tag metadata: %v", err)
	}
	
	redactedText, _, err := session.redact(text, tagMetadata)
	if err != nil {
		return "", fmt.Errorf("error redacting text: %v", err)
	}

	return redactedText, nil
}

func (session *ChatSession) Restore(text string) (string, error) {
	tagMetadata, err := session.getTagMetadata()
	if err != nil {
		return "", fmt.Errorf("error getting tag metadata: %v", err)
	}

	for replacement, original := range tagMetadata.TagMap {
		text = strings.ReplaceAll(text, replacement, original)
	}

	return text, nil
}

type ChatItem struct {
	RedactedText string
	Reply        string
	TagMap       map[string]string
}
type ChatIterator func(yield func(ChatItem, error) bool)

func (session *ChatSession) ChatStream(userInput string) (ChatIterator, error) {
	tagMetadata, err := session.getTagMetadata()
	if err != nil {
		return nil, fmt.Errorf("error getting tag metadata: %v", err)
	}

	redactedText, newTagMetadata, err := session.redact(userInput, tagMetadata)
	if err != nil {
		return nil, fmt.Errorf("error redacting user input: %v", err)
	}

	if err := session.updateTagMetadata(newTagMetadata); err != nil {
		return nil, fmt.Errorf("error updating tag map: %v", err)
	}

	history, err := session.getChatHistory()
	if err != nil {
		return nil, fmt.Errorf("error getting chat history: %v", err)
	}

	context := ""
	for _, msg := range history {
		context += fmt.Sprintf("%s: %s\n", msg.MessageType, msg.Content)
	}
	context += fmt.Sprintf("User: %s\n", redactedText)

	iterator := func(yield func(ChatItem, error) bool) {
		// First yield the non-streaming components
		// TODO: Will this the tag map get too big? Should we only yield
		// the subset of the tag map that is relevant to the current message?
		if !yield(ChatItem{RedactedText: redactedText, TagMap: newTagMetadata.TagMap}, nil) {
			log.Printf("Failed to yield redacted text and tag map")
			return
		}

		// Then stream the OpenAI response
		openaiResp := ""
		for chunk, err := range session.streamOpenAIResponse(context) {
			if err != nil {
				yield(ChatItem{}, err)
				return
			}
			openaiResp += chunk
			yield(ChatItem{Reply: chunk}, nil)
		}

		// Only save messages if the whole process was successful.
		// This gives the illusion of atomicity; a request either succeeds or fails entirely.
		// We still yield the error so the frontend can process accordingly.
		if err := session.saveMessage("user", redactedText, nil); err != nil {
			yield(ChatItem{}, err)
			return
		}

		if err := session.saveMessage("ai", openaiResp, nil); err != nil {
			yield(ChatItem{}, err)
			return
		}
	}

	return iterator, nil
}

func (session *ChatSession) getTagMetadata() (TagMetadata, error) {
	chatSession, err := session.db.GetSession(session.sessionID)
	if err != nil {
		return TagMetadata{}, err
	}

	var tagMetadata TagMetadata
	if err := json.Unmarshal(chatSession.TagMetadata, &tagMetadata); err != nil {
		return TagMetadata{}, err
	}

	return tagMetadata, nil
}

func (session *ChatSession) updateTagMetadata(tagMetadata TagMetadata) error {
	tagMetadataJSON, err := json.Marshal(tagMetadata)
	if err != nil {
		return fmt.Errorf("error marshalling tag map: %v", err)
	}
	return session.db.UpdateSessionTagMetadata(session.sessionID, tagMetadataJSON)
}

func (session *ChatSession) streamOpenAIResponse(ctx string) func(yield func(string, error) bool) {
	return func(yield func(string, error) bool) {
		messages := []llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeSystem, "You are a helpful assistant. Note that some fields will be obfuscated (e.g. [PERSON_1]) and will be injected back when it is displayed to the reader, so pretend like you know what they are. Additionally, each obfuscated token is a single word. E.g. 'John Doe' will be obfuscated as '[NAME_1] [NAME_2]' and 'Apple Corp' will be obfuscated as '[COMPANY_1] [COMPANY_2]'. You do not have to obfuscate your responses."),
			llms.TextParts(llms.ChatMessageTypeHuman, ctx),
		}

		_, err := session.openAIClient.GenerateContent(context.Background(), messages, llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			if !yield(string(chunk), nil) {
				return ErrStopStream
			}
			return nil
		}))
		if err != nil && !errors.Is(err, ErrStopStream) { // this might not be needed, but it might pass the error returned from the streaming func back here
			slog.Error("error during openai generation", "error", err)
			yield("", err) // return doesn't matter since there are no more yield calls
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
	return session.db.SaveChatMessage(&chatMessage)
}

func (session *ChatSession) getChatHistory() ([]database.ChatHistory, error) {
	return session.db.GetChatHistory(session.sessionID)
}
