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

  const getChatPreviews = async (): Promise<ChatPreview[]> => {
    const sessions = await nerService.getChatSessions();
    console.log(sessions);
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

    return history.data.map((message) => ({
      id: message.id,
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
    const invalidApiKey = () => false;
    if (invalidApiKey()) {
      setMessages(prevMessages);
      return {
        invalidApiKey: true,
        unsentMessage: message,
        errorMessage: 'Invalid API key',
      };
    }

    const otherError = () => false;
    if (otherError()) {
      setMessages(prevMessages);
      return {
        invalidApiKey: false,
        unsentMessage: message,
        errorMessage: 'Other error',
      };
    }

    setMessages([
      ...prevMessages,
      {
        id: `m-${prevMessages.length + 1}`,
        content: message,
        role: 'user',
      },
      {
        id: `m-${prevMessages.length + 2}`,
        content: `What is Lorem Ipsum?Lorem Ipsum is What is Lorem Ipsum?
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

        Why do we use it?
        e so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain. These cases are perfectly simple and easy to distinguish. In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided. But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted. The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains."simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.`,
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
    // TODO: Implement
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
  };
}
