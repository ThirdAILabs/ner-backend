import React, { useState, useRef, useEffect, ChangeEvent } from 'react';
import { MessageSquare, Paperclip } from 'lucide-react';
import { HiChip } from 'react-icons/hi';
import useOutsideClick from '@/hooks/useOutsideClick';
import { Message, RedactedContentPiece } from '@/hooks/useSafeGPT';
import Options from './Options';
import Markdown from 'react-markdown';
import './markdown.css';
import extractFileText from './extractFileText';

const NICE_COLOR_PAIRS = [
  { replacement: '#E57373', original: '#FFEBEE' }, // red
  { replacement: '#7986CB', original: '#E8EAF6' }, // indigo
  { replacement: '#4FC3F7', original: '#E1F5FE' }, // light blue
  { replacement: '#81C784', original: '#E8F5E9' }, // green
  { replacement: '#FFF176', original: '#FFFDE7' }, // yellow
  { replacement: '#FFB74D', original: '#FFF3E0' }, // orange
  { replacement: '#BA68C8', original: '#F3E5F5' } // purple
];

const TAG_COLORS: Record<string, { replacement: string; original: string }> =
  {};

const toTagName = (replacementToken: string) => {
  const pieces = replacementToken.split('_');
  return pieces.slice(0, pieces.length - 1).join('_');
};

const getTagColors = (tagName: string) => {
  if (!TAG_COLORS[tagName]) {
    const colorIndex = Object.keys(TAG_COLORS).length % NICE_COLOR_PAIRS.length;
    TAG_COLORS[tagName] = NICE_COLOR_PAIRS[colorIndex];
  }
  return TAG_COLORS[tagName];
};

function RedactedMessage({
  redactedContent
}: {
  redactedContent: RedactedContentPiece[];
}) {
  // TODO: Fix markdown rendering in redacted mode
  return (
    <div>
      {redactedContent.map((piece, idx) => {
        if (!piece.replacement) {
          return piece.original;
        }

        const tagName = toTagName(piece.replacement);
        const { replacement: replColor, original: origColor } =
          getTagColors(tagName);

        return (
          <span
            key={idx}
            className={`inline-flex items-center gap-1 p-1 pl-2 rounded-md`}
            style={{ backgroundColor: origColor }}
          >
            <del>{piece.original}</del>
            <span
              className={`px-1 rounded-sm text-white`}
              style={{ backgroundColor: replColor }}
            >
              {piece.replacement}
            </span>
          </span>
        );
      })}
    </div>
  );
}

interface ChatInterfaceProps {
  onSendMessage?: (message: string) => Promise<void>;
  messages: Message[];
  isLoading?: boolean;
  invalidApiKey: boolean;
  apiKey: string;
  saveApiKey: (key: string) => void;
  showRedaction: boolean;
  model: 'gpt-4o-mini' | 'gpt-4o';
  onSelectModel: (model: 'gpt-4o-mini' | 'gpt-4o') => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onSendMessage,
  messages,
  isLoading = false,
  invalidApiKey,
  apiKey,
  saveApiKey,
  showRedaction,
  model,
  onSelectModel
}) => {
  // Options menu-related logic
  const [isOptionsOpen, setIsOptionsOpen] = useState(false);
  const [editingApiKey, setEditingApiKey] = useState<boolean>(false);

  const openOptions = () => {
    setIsOptionsOpen(true);
  };

  const closeOptions = () => {
    setIsOptionsOpen(false);
    setEditingApiKey(false);
  };

  const closeOptionsIfNotEditing = () => {
    if (!editingApiKey) {
      setIsOptionsOpen(false);
    }
  };

  const optionsRef = useOutsideClick(() => {
    closeOptionsIfNotEditing();
  });

  useEffect(() => {
    if (invalidApiKey) {
      openOptions();
      setEditingApiKey(true);
    }
  }, [invalidApiKey]);

  const handleSaveApiKey = (key: string) => {
    saveApiKey(key);
    closeOptions();
  };

  const handleCancelApiKey = () => {
    closeOptions();
  };

  const handleEditApiKey = () => {
    setEditingApiKey(true);
  };

  // Chat-related logic

  const [isDragging, setIsDragging] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [justUploaded, setJustUploaded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const fileInputRef = useRef<HTMLInputElement>(null);

  // TODO: Why is there so much duplicate code here?
  const handleFileSelect = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    let extractedText = '';

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const text = await extractFileText(file);
        extractedText +=
          (extractedText ? '\n\n' : '') + `[From ${file.name}]:\n` + text;
      } catch (error) {
        console.error(`Failed to extract text from ${file.name}:`, error);
      }
    }

    if (extractedText) {
      setInputMessage((prev) => prev + (prev ? '\n\n' : '') + extractedText);
      setJustUploaded(true);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    let extractedText = '';

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      try {
        const text = await extractFileText(file);
        extractedText +=
          (extractedText ? '\n\n' : '') + `[From ${file.name}]:\n` + text;
      } catch (error) {
        console.error(`Failed to extract text from ${file.name}:`, error);
      }
    }

    if (extractedText) {
      setInputMessage((prev) => prev + (prev ? '\n\n' : '') + extractedText);
      setJustUploaded(true);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMessage.trim() && onSendMessage) {
      // Replace newlines with two spaces and a newline to get better
      // markdown formatting.
      onSendMessage(inputMessage.replaceAll('\n', '  \n')).catch((error) => {
        // Set input message to the last message that was sent
        // if send message fails.
        setInputMessage(inputMessage);
      });
      setInputMessage('');

      if (textareaRef.current) {
        textareaRef.current.style.height = '56px';
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const adjustTextareaHeight = (element: HTMLTextAreaElement) => {
    element.style.height = 'auto';
    element.style.height = Math.min(element.scrollHeight, 150) + 'px';
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputMessage(e.target.value);
  };

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'instant' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, showRedaction]);

  useEffect(() => {
    if (textareaRef.current) {
      adjustTextareaHeight(textareaRef.current);
      if (justUploaded) {
        textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
        setJustUploaded(false);
      }
    }
  }, [inputMessage, justUploaded]);

  return (
    <div
      className="flex flex-col h-[100%] relative w-full"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragging && (
        <div className="absolute inset-0 bg-white/90 flex flex-col items-center justify-center z-50 text-gray-500">
          <h3 className="text-xl font-semibold mb-2">Drop files here</h3>
          <p className="text-sm text-gray-400">(PDF only)</p>
        </div>
      )}
      <div className="flex-1 overflow-y-auto space-y-4 mb-20 p-4 px-[16%]">
        {messages.length === 0 && (
          <div className="h-[calc(100%-2rem)] flex flex-col items-center justify-center text-gray-500">
            <MessageSquare size={48} className="mb-4 text-gray-400" />
            <h3 className="text-xl font-semibold mb-2">
              Welcome to Secure Chat
            </h3>
            <p className="text-sm text-gray-400">
              Your messages are end-to-end encrypted and securely stored
            </p>
          </div>
        )}
        {messages.length !== 0 &&
          messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={` rounded-xl p-3 ${
                  message.role === 'user'
                    ? 'bg-gray-100 text-gray-700 p-6 max-w-[70%]'
                    : 'text-gray-600 text-lg/8 mt-6'
                } leading-relaxed`}
              >
                {!message.content && (
                  <div className="flex gap-1">
                    <div className="h-2 w-2 bg-gray-300 rounded-full animate-[pulse_1s_ease-in-out_infinite]"></div>
                    <div className="h-2 w-2 bg-gray-300 rounded-full animate-[pulse_1s_ease-in-out_infinite_0.2s]"></div>
                    <div className="h-2 w-2 bg-gray-300 rounded-full animate-[pulse_1s_ease-in-out_infinite_0.4s]"></div>
                  </div>
                )}
                {!showRedaction ? (
                  <div className="markdown-content">
                    <Markdown>{message.content}</Markdown>
                  </div>
                ) : (
                  <RedactedMessage redactedContent={message.redactedContent} />
                )}
              </div>
            </div>
          ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="absolute bottom-0 left-0 right-0 py-4 px-[14%]">
        <form onSubmit={handleSubmit} className="flex gap-2 relative">
          <textarea
            ref={textareaRef}
            value={inputMessage}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyPress}
            placeholder="Type a message..."
            rows={1}
            className="flex-1 p-4 pr-[85px] border-[1px] rounded-2xl resize-none min-h-[56px] max-h-[150px]"
            style={{
              overflowY:
                inputMessage &&
                textareaRef.current &&
                textareaRef.current.scrollHeight > 150
                  ? 'auto'
                  : 'hidden'
            }}
            disabled={isLoading}
          />
          <>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              multiple
              accept=".pdf,.docx,.xlsx,.pptx,.txt,.csv"
              className="hidden"
            />
            <button
              onClick={(e) => {
                e.preventDefault();
                fileInputRef.current?.click();
              }}
              className="absolute right-[55px] top-1/2 -translate-y-1/2 p-2 text-[rgb(85,152,229)] hover:text-[rgb(85,152,229)]/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Paperclip />
            </button>
          </>
          <div className="relative" ref={optionsRef}>
            <button
              disabled={isLoading}
              onClick={(e) => {
                e.preventDefault();
                if (isOptionsOpen) {
                  closeOptionsIfNotEditing();
                } else {
                  openOptions();
                }
              }}
              className="absolute right-4 top-1/2 -translate-y-1/2 p-2 text-[rgb(85,152,229)] hover:text-[rgb(85,152,229)]/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <HiChip size={22} />
            </button>
            {isOptionsOpen && (
              <div className="absolute bottom-12 right-4 w-[350px]">
                <Options
                  handleBasicMode={closeOptionsIfNotEditing}
                  handleAdvancedMode={() => {}}
                  model={model}
                  onSelectModel={onSelectModel}
                  apiKey={apiKey}
                  invalidApiKey={invalidApiKey}
                  onEditApiKey={handleEditApiKey}
                  onSaveApiKey={handleSaveApiKey}
                  onCancelApiKey={handleCancelApiKey}
                />
              </div>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
