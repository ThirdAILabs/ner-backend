'use client';

import { useEffect, useRef, useState } from 'react';
import { nerService } from '@/lib/backend';

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

export default function useSafeGPT(chatId: string) {
  const [title, setTitle] = useState('Loading...');
  const [previews, setPreviews] = useState<ChatPreview[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [invalidApiKey, setInvalidApiKey] = useState<boolean>(false);
  
  const getChatPreviews = async (): Promise<ChatPreview[]> => {
    const sessions = await nerService.getChatSessions();
    if (sessions.error) {
      alert(sessions.error);
      return [];
    }

    return sessions.data
      .map((session) => ({
        id: session.id,
        title: session.title,
      }))
      .reverse(); // reverse to show most recent chats first TODO this should be handled by backend and based on last modified not last created.
  };

  const getChat = async (chatId: string): Promise<Message[]> => {
    if (chatId === NEW_CHAT_ID) {
      setTitle('New Chat');
      return [];
    }

    const session = await nerService.getChatSession(chatId);
    if (session.error) {
      alert(session.error);
      return [];
    }

    setTitle(session.data!.title || `Chat ${chatId}`);
    const tagMap = session.data!.tag_map;

    const history = await nerService.getChatHistory(chatId);

    if (!history.data) {
      return [];
    }

    const messages: Message[] = [];
    for (const message of history.data) {
      messages.push({
        content: unredactContent(message.content, tagMap),
        redactedContent: toRedactedContent(message.content, tagMap),
        role: message.message_type === 'user' ? 'user' : 'llm',
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
    if (selectedId === chatId) {
      window.location.href = `/safegpt?id=new`;
    }
  };

  const sendMessage = async (message: string, apiKey: string): Promise<void> => {
    setInvalidApiKey(false);

    let sessionId = chatId;
    if (chatId === NEW_CHAT_ID) {
      const { data } = await nerService.startChatSession('gpt-4', title);
      if (data?.session_id) {
        sessionId = data.session_id;
      }
    }

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

    try {
      await nerService.sendChatMessageStream(sessionId, 'gpt-4', apiKey, message, (chunk) => {
        if (chunk.tag_map) {
          tagMap = { ...tagMap, ...chunk.tag_map };
        }

        if (chunk.input_text) {
          setMessages([
            ...prevMessages,
            {
              content: unredactContent(chunk.input_text, tagMap),
              redactedContent: toRedactedContent(chunk.input_text, tagMap),
              role: 'user',
            },
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
        if (chunk.reply) {
          replyBuilder += chunk.reply;
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
      });
      if (sessionId !== chatId) {
        window.location.href = `/safegpt?id=${sessionId}`;
      }
    } catch (error) {
      const errorMessage = (error as Error).message;
      if (
        errorMessage.includes('Incorrect API key') ||
        errorMessage.includes('missing the OpenAI API key')
      ) {
        setMessages(prevMessages);
        setInvalidApiKey(true);
        if (sessionId !== chatId) {
          deleteChat(sessionId);
        }
      } else {
        setMessages(prevMessages);
        alert(errorMessage);
        if (sessionId !== chatId) {
          deleteChat(sessionId);
        }
      }
      throw new Error(errorMessage);
    }
  };

  const updateTitle = async (newTitle: string) => {
    let sessionId = chatId;
    if (chatId === NEW_CHAT_ID) {
      const { data } = await nerService.startChatSession('gpt-4', newTitle);
      if (data?.session_id) {
        sessionId = data.session_id;
      }
    }
    await nerService.renameChatSession(sessionId, newTitle);
    setTitle(newTitle);
    setPreviews((prevPreviews) =>
      prevPreviews.map((preview) =>
        preview.id === chatId ? { ...preview, title: newTitle } : preview
      )
    );

    if (sessionId !== chatId) {
      window.location.href = `/safegpt?id=${sessionId}`;
    }
  };

  useEffect(() => {
    getChatPreviews().then(setPreviews);
  }, []);

  useEffect(() => {
    getChat(chatId).then(setMessages);
  }, [chatId]);

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
