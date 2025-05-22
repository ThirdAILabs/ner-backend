'use client';

import { ChatPreview } from '@/components/chat/Sidebar';
import { useEffect, useState } from 'react';
import { Message } from '@/components/chat/Chat';

export default function useSafeGPT(chatId: string) {
  const [title, setTitle] = useState('Loading...');
  const [previews, setPreviews] = useState<ChatPreview[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);

  const getChatPreviews = async (): Promise<ChatPreview[]> => {
    // TODO: Implement
    return Array.from({ length: 101 }, (_, i) => ({
      id: i.toString(),
      title: `Chat ${i}`,
    }));
  };

  const getChat = async (chatId: string): Promise<Message[]> => {
    // TODO: Implement
    if (chatId === 'new') {
      setTitle('New chat');
      return [];
    }
    setTitle(`Chat ${chatId}`);
    return [
      {
        id: 'm-2',
        content: `What is Lorem Ipsum?Lorem Ipsum is What is Lorem Ipsum?
        Popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.`,
        role: 'user',
      },
      {
        id: 'm-1',
        content: `What is Lorem Ipsum?Lorem Ipsum is What is Lorem Ipsum?
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

        Why do we use it?
        e so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain. These cases are perfectly simple and easy to distinguish. In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided. But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted. The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains."simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.`,
        role: 'llm',
      },
    ];
  };

  const sendMessage = async (message: string) => {
    // TODO: Implement
    // TODO: If id is new, create a new chat history, append to previews.
    setMessages((prev) => [
      ...prev,
      {
        id: `m-${prev.length + 1}`,
        content: message,
        role: 'user',
      },
      {
        id: `m-${prev.length + 2}`,
        content: `What is Lorem Ipsum?Lorem Ipsum is What is Lorem Ipsum?
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

        Why do we use it?
        e so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain. These cases are perfectly simple and easy to distinguish. In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided. But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted. The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains."simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.`,
        role: 'llm',
      },
    ]);
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
