'use client';

import { ChatPreview } from '@/components/chat/Sidebar';
import { useEffect, useState } from 'react';
import { Message } from '@/components/chat/Chat';
import { nerService } from '@/lib/backend';

interface SendMessageResponse {
  invalidApiKey: boolean;
  unsentMessage: string;
  errorMessage: string;
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
    if (chatId === 'new') {
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

    return history.data.map((message, idx) => ({
      id: `m-${idx}`,
      content: message.content,
      role: message.message_type === 'user' ? 'user' : 'llm',
    }));
  };

  const sendMessage = async (message: string, apiKey: string): Promise<SendMessageResponse> => {
    const prevMessages = messages;
    setMessages([
      ...prevMessages,
      {
        id: `m-${prevMessages.length + 1}`,
        content: message,
        role: 'user',
      },
    ]);

    const response = await nerService.sendChatMessage(chatId, "gpt-4", message, apiKey);
    
    if (response.error && response.error.includes('could not create OpenAI client')) {
      setMessages(prevMessages);
      return {
        invalidApiKey: false,
        unsentMessage: message,
        errorMessage: response.error,
      };
    }

    if (response.error) {
      setMessages(prevMessages);
      alert(response.error);
      return {
        invalidApiKey: false,
        unsentMessage: message,
        errorMessage: response.error,
      };
    }

    // TODO: Set messages to ...prevMessages, response
    setMessages([
      ...prevMessages,
      {
        id: `m-${prevMessages.length + 1}`,
        content: message,
        role: 'user',
      },
      {
        id: `m-${prevMessages.length + 2}`,
        content: response.data?.reply || '',
        role: 'llm',
      },
    ])

    return {
      invalidApiKey: false,
      unsentMessage: '',
      errorMessage: '',
    }
  };

  const updateTitle = (title: string) => {
    // TODO: Implement.
    setTitle(title);
  };

  useEffect(() => {
    getChatPreviews().then(setPreviews);
  }, []);

  useEffect(() => {
    getChat(chatId).then(setMessages);
  }, [chatId]);

  return {
    previews: [
      ...(chatId === 'new'
        ? [
            {
              id: 'new',
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
