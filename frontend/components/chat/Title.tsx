import { useState, useEffect } from 'react';
import SaveAndCancel from './SaveAndCancel';

export interface ChatTitleProps {
  title: string;
  setTitle: (title: string) => void;
}

function EditButton({ onClick }: { onClick: () => void }) {
  return (
    <button onClick={onClick} className="text-gray-400 hover:text-gray-600">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
      </svg>
    </button>
  );
}

export default function ChatTitle({ title, setTitle }: ChatTitleProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(title);

  useEffect(() => {
    setInputValue(title);
  }, [title]);

  return (
    <div className="flex items-center w-full px-4 relative">
      <div className="flex-1 flex justify-center">
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            {isEditing ? (
              <>
                <div className="relative inline-block">
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    className="text-xl font-medium bg-transparent border-none border-b-2 border-gray-300 focus:outline-none focus:border-blue-500 transition-all w-full text-center"
                    style={{ width: `${inputValue.length}ch` }}
                    autoFocus
                  />
                </div>
                <SaveAndCancel
                  onSave={() => {
                    setIsEditing(false);
                    setTitle(inputValue);
                  }}
                  onCancel={() => {
                    setIsEditing(false);
                    setInputValue(title);
                  }}
                />
              </>
            ) : (
              <>
                {/* This div is the same width as the edit button to center the title */}
                <div className="w-[16px] text-transparent">.</div>
                <span className="text-xl font-medium" onClick={() => setIsEditing(true)}>
                  {title}
                </span>
                <EditButton onClick={() => setIsEditing(true)} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
