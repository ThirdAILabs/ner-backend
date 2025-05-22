import React, { useState, useRef, useEffect } from 'react';
import { Send, MessageSquare } from 'lucide-react';
import { HiChip } from 'react-icons/hi';
import useOutsideClick from '@/hooks/useOutsideClick';
import Options from './Options';

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'llm';
}

interface ChatInterfaceProps {
  onSendMessage?: (message: string) => void;
  messages: Message[];
  isLoading?: boolean;
  invalidApiKey: boolean;
  apiKey: string;
  setApiKey: (key: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onSendMessage,
  messages,
  isLoading = false,
  invalidApiKey,
  apiKey,
  setApiKey,
}) => {
  // Options dropdown-related logic
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [editingApiKey, setEditingApiKey] = useState<boolean>(false);

  const openDropdown = () => {
    setIsDropdownOpen(true);
  };

  const closeDropdown = () => {
    setIsDropdownOpen(false);
    setEditingApiKey(false);
  };

  const closeDropdownIfNotEditing = () => {
    if (!editingApiKey) {
      setIsDropdownOpen(false);
    }
  };

  const dropdownRef = useOutsideClick(() => {
    closeDropdownIfNotEditing();
  });

  useEffect(() => {
    if (invalidApiKey) {
      openDropdown();
      setEditingApiKey(true);
    }
  }, [invalidApiKey]);

  const handleSaveApiKey = (key: string) => {
    setApiKey(key);
    closeDropdown();
  };

  const handleCancelApiKey = () => {
    closeDropdown();
  };

  const handleEditApiKey = () => {
    setEditingApiKey(true);
  };
  
  // Chat-related logic

  const [inputMessage, setInputMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputMessage.trim() && onSendMessage) {
      onSendMessage(inputMessage);
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
    adjustTextareaHeight(e.target);
  };
  /*
    Todo:-
    1. Css for llm message should be good.
    2. Correct the "Your messages are end-to-end encrypted and securely stored" message.
     */
  
  return (
    <div className="flex flex-col h-[100%] relative w-[80%] ml-[10%]">
      <div className="flex-1 overflow-y-auto p-4 space-y-4 mb-20">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-gray-500">
            <MessageSquare size={48} className="mb-4 text-gray-400" />
            <h3 className="text-xl font-semibold mb-2">Welcome to Secure Chat</h3>
            <p className="text-sm text-gray-400">
              Your messages are end-to-end encrypted and securely stored
            </p>
          </div>
        )}
        {messages.length !== 0 &&
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={` rounded-xl p-3 ${
                  message.role === 'user'
                    ? 'bg-gray-100 text-gray-700 p-6 max-w-[70%]'
                    : 'text-gray-600 text-lg/8 mt-6'
                } leading-relaxed`}
              >
                {message.content}
              </div>
            </div>
          ))}
      </div>

      <div className="absolute bottom-0 left-0 right-0 py-4">
        <form onSubmit={handleSubmit} className="flex gap-2 relative">
          <textarea
            ref={textareaRef}
            value={inputMessage}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyPress}
            placeholder="Type a message..."
            rows={1}
            className="flex-1 p-4 pr-16 border-[1px] rounded-2xl resize-none min-h-[56px] max-h-[150px] overflow-y-auto"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!inputMessage.trim() || isLoading}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-[rgb(85,152,229)] hover:text-[rgb(85,152,229)]/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={20} />
          </button>
          <div className="relative" ref={dropdownRef}>
            <button
              disabled={isLoading}
              onClick={() => {
                if (isDropdownOpen) {
                  closeDropdownIfNotEditing();
                } else {
                  openDropdown();
                }
              }}
              className="absolute right-10 top-1/2 -translate-y-1/2 p-2 text-[rgb(85,152,229)] hover:text-[rgb(85,152,229)]/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <HiChip size={22} />
            </button>
            {isDropdownOpen && (
              <div className="absolute bottom-12 right-0 w-[350px]">
                <Options
                  handleBasicMode={closeDropdownIfNotEditing}
                  handleAdvancedMode={() => {}}
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
