import React, { useState, useRef, useEffect, ChangeEvent } from 'react';
import { Send, MessageSquare, Lock } from 'lucide-react';
import { HiChip } from 'react-icons/hi';
import useOutsideClick from '@/hooks/useOutsideClick';
import { Tooltip, Typography } from '@mui/material';
import { CardTitle } from '../ui/card';
import SaveAndCancel from './SaveAndCancel';


interface ModelOptionProps {
  children: React.ReactNode;
  onClick: () => void;
  selected: boolean;
  disabled?: boolean;
}

function ModelOption({ children, onClick, selected, disabled }: ModelOptionProps) {
  const selectedStyle = "bg-[rgb(85,152,229)]/10 text-[rgb(85,152,229)] font-semibold border-[rgb(85,152,229)]";
  const unselectedStyle = "text-gray-700 border-gray-200";
  const disabledStyle = "cursor-not-allowed";
  const enabledStyle = "hover:bg-[rgb(85,152,229)]/5";
  return <button
    type="button"
    className={
      `inline w-full rounded-sm px-4 py-2 text-sm border text-left transition-colors flex items-center gap-2
      ${ selected ? selectedStyle : unselectedStyle}
      ${disabled ? disabledStyle : enabledStyle}`
    }
    onClick={onClick}
  >
    {children}
  </button>
}

interface OptionsDropdownProps {
  handleBasicMode: () => void;
  handleAdvancedMode: () => void;
  apiKey: string;
  invalidApiKey: boolean;
  editingApiKey: boolean;
  onEditApiKey: () => void;
  onSaveAPIKey: (key: string) => void;
  onCancelAPIKey: () => void;
}

function OptionsDropdown({ handleBasicMode, handleAdvancedMode, apiKey, invalidApiKey, editingApiKey, onEditApiKey, onSaveAPIKey, onCancelAPIKey }: OptionsDropdownProps) {
  const apiKeyRef = useRef<HTMLInputElement>(null);
  const [intAPIKey, setIntAPIKey] = useState<string>("");
  const [showInvalidKeyError, setShowInvalidKeyError] = useState<boolean>(invalidApiKey);

  useEffect(() => {
    setIntAPIKey(apiKey);
    if (apiKey === "") {
      setShowInvalidKeyError(true);
    }
  }, [apiKey]);

  useEffect(() => {
    setShowInvalidKeyError(invalidApiKey);
  }, [invalidApiKey]);

  useEffect(() => {
    if (editingApiKey) {
      apiKeyRef.current?.focus();
    } else {
      apiKeyRef.current?.blur();
    }
  }, [editingApiKey]);

  const handleSaveApiKey = () => {
    if (intAPIKey === "") {
      setShowInvalidKeyError(true);
    } else {
      onSaveAPIKey(intAPIKey);
      console.log("Blurring...");
      apiKeyRef.current?.blur();
    }
  }

  const handleCancelApiKey = () => {
    if (apiKey !== "") {
      setIntAPIKey(apiKey);
      onCancelAPIKey();
      setShowInvalidKeyError(false);
    } else {
      setShowInvalidKeyError(true);
    }
  }

  const handleApiKeyChange = (e: ChangeEvent<HTMLInputElement>) => {
    onEditApiKey();
    setIntAPIKey(e.target.value);
    setShowInvalidKeyError(false);
  } 

  return (
    <div className="absolute bottom-12 right-0 w-[350px] bg-white rounded-md shadow-lg border border-gray-200">
      <div className="p-2 flex flex-col gap-1">
        <div className="inline font-semibold p-1">
          Redaction Model
        </div>

        <div className="flex flex-row w-full gap-2">
          <div className="w-full">
            <ModelOption onClick={handleBasicMode} selected={true}>
              <MessageSquare className="w-4 h-4" />
              <span>Basic</span>
            </ModelOption>
          </div>
          <div className="w-full">
            <Tooltip title="Requires pro subscription. Email us at contact@thirdai.com">
              {/* Need span because tooltip child cannot be a custom react component. */}
              <span className="w-full">
                <ModelOption onClick={handleAdvancedMode} selected={false} disabled>
                  <Lock className="w-4 h-4" />
                  <span>Advanced</span>
                </ModelOption>
              </span>
            </Tooltip>
          </div>
        </div>

        <div className="inline font-semibold p-1">
          OpenAI Key
        </div>
        <div className="flex flex-row w-full gap-2">
          <input
            type="text"
            ref={apiKeyRef}
            value={intAPIKey}
            onChange={handleApiKeyChange}
            onFocus={() => {
              console.log("Focusing...");
            }}
            onBlur={() => {
              if (editingApiKey) {
                console.log("editingApiKey is", editingApiKey);
                apiKeyRef.current?.focus();
              }
            }}
            placeholder="Your API key here..."
            className="flex-1 px-4 py-2 min-w-[100px] border-[1px] rounded-md"
          />
          {
            editingApiKey && (
              <SaveAndCancel onSave={handleSaveApiKey} onCancel={handleCancelApiKey} />
            )
          }
        </div>
        <div className="block h-5 text-red-500 text-sm p-1 pb-5">
          {
            showInvalidKeyError ?
              "The OpenAI key is invalid." :
              ""
          }
        </div>
      </div>
    </div>
  );
}

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
  setAPIKey: (key: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onSendMessage,
  messages,
  isLoading = false,
  invalidApiKey,
  apiKey,
  setAPIKey,
}) => {
  const [editingApiKey, setEditingApiKey] = useState<boolean>(false);
  const [inputMessage, setInputMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  console.log("Editing API key: ", editingApiKey);
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
  console.log('Input message...', inputMessage);
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
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useOutsideClick(() => {
    handleCloseDropdown();
  });

  useEffect(() => {
    if (invalidApiKey) {
      setIsDropdownOpen(true);
      setEditingApiKey(true);
    }
  }, [invalidApiKey]);

  const handleCloseDropdown = () => {
    if (!editingApiKey) {
      setIsDropdownOpen(false);
    }
  }

  const handleSaveApiKey = (key: string) => {
    setAPIKey(key);
    setEditingApiKey(false);
  };

  const handleCancelApiKey = () => {
    if (apiKey !== "") {
      setEditingApiKey(false);
    }
  }

  const handleEditApiKey = () => {
    setEditingApiKey(true);
  }

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
              onClick={() => setIsDropdownOpen((prev) => {
                if (!prev) {
                  return true;
                }
                if (!editingApiKey) {
                  return false;
                }
                return prev;
              })}
              className="absolute right-10 top-1/2 -translate-y-1/2 p-2 text-[rgb(85,152,229)] hover:text-[rgb(85,152,229)]/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <HiChip size={22} />
            </button>
              {isDropdownOpen && (
                <OptionsDropdown
                  handleBasicMode={() => {
                    // Handle basic mode
                    if (!editingApiKey) {
                      setIsDropdownOpen(false);
                    }
                  }}
                  handleAdvancedMode={() => {}}
                  apiKey={apiKey}
                  invalidApiKey={invalidApiKey}
                  editingApiKey={editingApiKey}
                  onEditApiKey={handleEditApiKey}
                  onSaveAPIKey={handleSaveApiKey}
                  onCancelAPIKey={handleCancelApiKey}
                />
              )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
