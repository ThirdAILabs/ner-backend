'use client';

import { useEffect, useRef, useState } from 'react';
import { ChatResponse, nerService } from '@/lib/backend';

export interface ChatPreview {
  id: string;
  title: string;
}

export interface RedactedContentPiece {
  original: string;
  replacement?: string;
}

export interface Message {
  content: string;
  redactedContent: RedactedContentPiece[];
  role: 'user' | 'llm';
}

export const NEW_CHAT_ID = 'new';

/**
 * Converts a string with redacted tokens into an array of spans of
 * RedactedContentPiece objects, which represent redacted and unredacted
 * spans of text for easier downstream rendering.
 *
 * For example, if the input is "Hello [NAME_1] and [NAME_2]!", and the tagMap is
 * { "[NAME_1]": "John", "[NAME_2]": "Jane" }, the output will be:
 * [
 *   { original: "Hello " },
 *   { original: "John", replacement: "[NAME_1]" },
 *   { original: " and " },
 *   { original: "Jane", replacement: "[NAME_2]" },
 * ]
 * @param redactedContent A string that contains 0 or more redacted tokens of the form [TAG_NAME_1], [TAG_NAME_2], etc.
 * @param tagMap A map of redacted tokens to their original values.
 * @returns An array of redacted content pieces.
 */
const toRedactedContent = (
  redactedContent: string,
  tagMap: Record<string, string>
): RedactedContentPiece[] => {
  const redactedPieces: RedactedContentPiece[] = [];
  const regex = /\[(.*?)_\d+\]/g;
  let lastEndIndex = 0;
  let match;

  while ((match = regex.exec(redactedContent)) !== null) {
    const replacement = match[0];
    const original = tagMap[replacement];
    const startIndex = match.index;

    // Add the content that precedes the redacted tokens.
    redactedPieces.push({ original: redactedContent.slice(lastEndIndex, startIndex) });

    if (!original) {
      // False positive. This is an instance of [...] that has nothing to do with redaction.
      // TODO: Make sure replacement token does not appear in the original text.
      redactedPieces.push({ original: replacement });
    } else {
      redactedPieces.push({ original, replacement });
    }

    lastEndIndex = startIndex + replacement.length;
  }

  if (lastEndIndex < redactedContent.length) {
    redactedPieces.push({ original: redactedContent.slice(lastEndIndex) });
  }

  return redactedPieces;
};

const unredactContent = (content: string, tagMap: Record<string, string>) => {
  let unredactedContent = content;
  for (const [replacement, original] of Object.entries(tagMap)) {
    unredactedContent = unredactedContent.replaceAll(replacement, original);
  }
  return unredactedContent;
};

/**
 * While backend.ts is responsible for handling the raw API calls and type definitions,
 * this hook manages the higher-level chat functionality - maintaining UI state,
 * coordinating message flows, and providing a clean interface for components to
 * interact with the chat system.
 * @param chatId The ID of the current chat session
 */
export default function useSafeGPT(chatId: string) {
  const [title, setTitle] = useState('Loading...');
  const [previews, setPreviews] = useState<ChatPreview[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [invalidApiKey, setInvalidApiKey] = useState<boolean>(false);

  useEffect(() => {
    getChatPreviews().then(setPreviews);
  }, [chatId]);

  useEffect(() => {
    getChat(chatId).then(setMessages);
  }, [chatId]);

  const getChatPreviews = async (): Promise<ChatPreview[]> => {
    try {
      const sessions = await nerService.getChatSessions();
      return sessions
        .map((session) => ({
          id: session.ID,
          title: session.Title,
        }))
        .reverse(); // reverse to show most recent chats first TODO this should be handled by backend and based on last modified not last created.
    } catch (error) {
      alert('Failed to get chat sessions. Please try again.');
      return [];
    }
  };

  const getChat = async (chatId: string): Promise<Message[]> => {
    if (chatId === NEW_CHAT_ID) {
      setTitle('New Chat');
      return [];
    }

    let session;
    try {
      session = await nerService.getChatSession(chatId);
    } catch (error) {
      alert('Failed to get chat. Please try again.');
      return [];
    }

    setTitle(session.Title || `Chat ${chatId}`);
    const tagMap = session.TagMap;

    let history;
    try {
      history = await nerService.getChatHistory(chatId);
    } catch (error) {
      alert('Failed to get chat history. Please try again.');
      return [];
    }

    const messages: Message[] = [];
    for (const message of history) {
      messages.push({
        content: unredactContent(message.Content, tagMap),
        redactedContent: toRedactedContent(message.Content, tagMap),
        role: message.MessageType === 'user' ? 'user' : 'llm',
      });
    }

    return messages;
  };

  const deleteChat = async (selectedId: string) => {
    if (selectedId === NEW_CHAT_ID) {
      return;
    }
    await nerService.deleteChatSession(selectedId);
    setPreviews((prevPreviews) => prevPreviews.filter((preview) => preview.id !== selectedId));
  };

  const sendMessage = async (message: string, apiKey: string) => {
    setInvalidApiKey(false);

    // If the chat is new, create a new chat session.
    let sessionId = chatId;
    if (chatId === NEW_CHAT_ID) {
      sessionId = await nerService.startChatSession('gpt-4', title);
    }

    // Save the current state of messages to restore if the request fails.
    // This creates an illusion of atomicity - the operation either succeeds completely
    // or fails without side effects by rolling back to the previous state.
    const prevMessages = messages;
    setMessages([
      ...prevMessages,
      {
        content: message,
        redactedContent: [{ original: message }],
        role: 'user',
      },
    ]);

    let tagMap: Record<string, string> = {};
    let replyBuilder: string = '';

    const handleChunk = (chunk: ChatResponse) => {
      if (chunk.TagMap) {
        tagMap = { ...tagMap, ...chunk.TagMap };
      }

      if (chunk.InputText) {
        setMessages([
          ...prevMessages,
          {
            content: unredactContent(chunk.InputText, tagMap),
            redactedContent: toRedactedContent(chunk.InputText, tagMap),
            role: 'user',
          },
          // Add a placeholder message for the LLM response so we don't have to
          // conditionally add it to the messages array when we receive the first chunk
          // of the LLM response. It also reflects the fact that we're actively
          // waiting for the LLM response, allowing us to show a loading state.
          // This will not result in partial messages because we always rollback if the
          // transaction fails.
          {
            content: '',
            redactedContent: [],
            role: 'llm',
          },
        ]);
        return;
      }

      // We assume that the input text is always the first message.
      // Thus, the last message must be the LLM message.
      if (chunk.Reply) {
        replyBuilder += chunk.Reply;
        setMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].content = unredactContent(replyBuilder, tagMap);
          newMessages[newMessages.length - 1].redactedContent = toRedactedContent(
            replyBuilder,
            tagMap
          );
          return newMessages;
        });
      }
    };

    try {
      await nerService.sendChatMessageStream(sessionId, 'gpt-4', apiKey, message, handleChunk);
    } catch (error) {
      if ((error as Error).message.toLowerCase().includes('api key')) {
        setInvalidApiKey(true);
      } else {
        alert(
          'Failed to send message to GPT. Please make sure your API key is correct and you are connected to the internet then try again.'
        );
      }

      setMessages(prevMessages);
      if (sessionId !== chatId) {
        deleteChat(sessionId);
      }
      throw error;
    }

    if (sessionId !== chatId) {
      return sessionId;
    }

    return null;
  };

  const updateTitle = async (newTitle: string) => {
    let sessionId = chatId;
    if (chatId === NEW_CHAT_ID) {
      sessionId = await nerService.startChatSession('gpt-4', newTitle);
    }

    try {
      await nerService.renameChatSession(sessionId, newTitle);
      setTitle(newTitle);
      setPreviews((prevPreviews) =>
        prevPreviews.map((preview) =>
          preview.id === chatId ? { ...preview, title: newTitle } : preview
        )
      );

      if (sessionId !== chatId) {
        return sessionId;
      }
    } catch (error) {
      alert('Failed to update chat title. Please try again.');
    }

    return null;
  };

  return {
    previews: [
      ...(chatId === NEW_CHAT_ID
        ? [
            {
              id: NEW_CHAT_ID,
              title: 'New Chat',
            },
          ]
        : []),
      ...previews,
    ],
    title,
    updateTitle,
    messages,
    sendMessage,
    invalidApiKey,
    deleteChat,
  };
}
