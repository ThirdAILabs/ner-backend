'use client';

import { ChatPreview } from '@/components/chat/Sidebar';
import { useEffect, useState } from 'react';
import { Message } from '@/components/chat/Chat';
import { nerService } from '@/lib/backend';

const NEW_CHAT_ID = "new";

const strikethrough = (text: string) => {
  return (
    text
      .split('')
      .map(char => char + '\u0336')
      .join('')
  );
}

const displayRedactedContent = (redactedContent: string, tagMap: Record<string, string>) => {
  let displayContent = redactedContent;
  for (const [replacement, original] of Object.entries(tagMap)) {
    displayContent = displayContent.replace(replacement, `<del>${original}</del>` + ' ' + replacement);
  }
  return displayContent;
}

const unredactContent = (content: string, tagMap: Record<string, string>) => {
  let unredactedContent = content;
  for (const [replacement, original] of Object.entries(tagMap)) {
    unredactedContent = unredactedContent.replace(replacement, original);
  }
  return unredactedContent;
}

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

    return sessions.data.map((session) => ({
      id: session.id,
      title: session.title,
    }));
  };

  const getChat = async (chatId: string): Promise<Message[]> => {
    if (chatId === NEW_CHAT_ID) {
      setTitle('New chat');
      return [];
    }

    const session = await nerService.getChatSession(chatId);
    if (session.error) {
      alert(session.error);
      return [];
    }

    setTitle(session.data?.title || `Chat ${chatId}`);

    const history = await nerService.getChatHistory(chatId);

    if (!history.data) {
      return [];
    }

    return history.data.map((message, idx) => ({
      id: `m-${idx}`,
      content: message.content,
      redactedContent: message.content, // TODO: get redacted content
      role: message.message_type === 'user' ? 'user' : 'llm',
    }));
  };

  const sendMessage = async (message: string, apiKey: string): Promise<void> => {
    setInvalidApiKey(false);

    let sessionId = chatId;
    if (chatId === NEW_CHAT_ID) {
      const { data } = await nerService.startChatSession("gpt-4", title);
      if (data?.session_id) {
        sessionId = data.session_id;
      }
    }
    
    const prevMessages = messages;
    setMessages([
      ...prevMessages,
      {
        id: `m-${prevMessages.length + 1}`,
        content: message,
        redactedContent: message,
        role: 'user',
      },
    ]);

    const response = await nerService.sendChatMessage(sessionId, "gpt-4", apiKey, message);
    
    if (response.error && response.error.includes('Incorrect API key')) {
      console.log("Invalid API key");
      setMessages(prevMessages);
      setInvalidApiKey(true);
    }

    if (response.error) {
      setMessages(prevMessages);
      alert(response.error);
    }

    // TODO: Set messages to ...prevMessages, response
    setMessages([
      ...prevMessages,
      {
        id: `m-${prevMessages.length + 1}`,
        content: message,
        redactedContent: displayRedactedContent(response.data?.input_text || message, response.data?.tag_map || {}),
        role: 'user',
      },
      {
        id: `m-${prevMessages.length + 2}`,
        content: unredactContent(response.data?.reply || '', response.data?.tag_map || {}),
        redactedContent: displayRedactedContent(response.data?.reply || '', response.data?.tag_map || {}),
        role: 'llm',
      },
    ])
  };

  const updateTitle = async (title: string) => {
    let sessionId = chatId;
    if (chatId === NEW_CHAT_ID) {
      const { data } = await nerService.startChatSession("gpt-4", title);
      if (data?.session_id) {
        sessionId = data.session_id;
      }
    }
    await nerService.renameChatSession(sessionId, title);
    setTitle(title);
    window.location.href = `/safegpt?id=${sessionId}`;
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
              title: 'New chat',
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
  };
}
