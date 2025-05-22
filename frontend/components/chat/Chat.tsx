import React, { useState, useRef } from 'react';
import { Send, MessageSquare, Lock } from 'lucide-react';
import { HiChip } from "react-icons/hi";
import useOutsideClick from '@/hooks/useOutsideClick';

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'llm';
}

interface ChatInterfaceProps {
  jobId?: string;
  onSendMessage?: (message: string) => void;
  messages?: Message[];
  isLoading?: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  jobId,
  onSendMessage,
  messages = [{
    id: "m-2",
    content: `What is Lorem Ipsum?Lorem Ipsum is What is Lorem Ipsum?
  pularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.`,
    role: "user",
  }, {
    id: "m-1",
    content: `What is Lorem Ipsum?Lorem Ipsum is What is Lorem Ipsum?
          Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

          Why do we use it?
          e so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain. These cases are perfectly simple and easy to distinguish. In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided. But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted. The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains."simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.`,
    role: "llm",
  }],
  isLoading = false,
  //   messages = [],
}) => {
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
    setIsDropdownOpen(false);
  });

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
                className={` rounded-xl p-3 ${message.role === 'user'
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
            className="absolute right-3 top-1/2 -translate-y-1/2 p-2 text-blue-500 hover:text-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send size={20} />
          </button>
          <div className="relative" ref={dropdownRef}>
            <button
              disabled={isLoading}
              onClick={() => setIsDropdownOpen(prev => !prev)}
              className="absolute right-10 top-1/2 -translate-y-1/2 p-2 text-blue-500 hover:text-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <HiChip size={22} />
            </button>
            {isDropdownOpen && (
              <div className="absolute bottom-12 right-0 w-48 bg-white rounded-md shadow-lg border border-gray-200">
                <div className="py-2">
                  <button
                    type="button"
                    className="w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 text-left transition-colors flex items-center gap-2"
                    onClick={() => {
                      // Handle basic mode
                      setIsDropdownOpen(false);
                    }}
                  >
                    <MessageSquare className="w-4 h-4" />
                    <span>Basic</span>
                  </button>
                  <div className="relative">
                    <button
                      type="button"
                      disabled
                      className="w-full px-4 py-2 text-sm text-gray-400 hover:bg-gray-50 text-left transition-colors flex items-center gap-2 cursor-not-allowed"
                      onClick={() => setIsDropdownOpen(false)}
                    >
                      <Lock className="w-4 h-4" />
                      <span>Advanced</span>
                    </button>
                    <div className="absolute invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-opacity bg-gray-800 text-white text-xs rounded py-1 px-2 right-full mr-2 top-1/2 -translate-y-1/2 w-48">
                      Requires pro subscription. Email us at contact@thirdai.com
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
