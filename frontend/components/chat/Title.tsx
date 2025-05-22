import { useState, useEffect } from "react";

export interface ChatTitleProps {
  title: string;
  setTitle: (title: string) => void;
  showRedaction: boolean;
  onToggleRedaction: () => void;
}

function EditButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="text-gray-400 hover:text-gray-600"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />
      </svg>
    </button>
  );
}

function SaveButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="p-1.5 rounded-full border border-black bg-black hover:border-[rgb(85,152,229)] hover:bg-[rgb(85,152,229)] text-white transition-colors"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 6L9 17l-5-5" />
      </svg>
    </button>
  );
}

function CancelButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="p-1.5 rounded-full border border-black text-black hover:border-red-500 hover:text-red-500 transition-colors"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
        <path d="M18 6L6 18M6 6l12 12" />
      </svg>
    </button>
  );
}

function Toggle({ checked, onChange }: { checked?: boolean; onChange?: () => void }) {
  return (
    <button
      onClick={onChange}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${checked ? 'bg-blue-600' : 'bg-gray-200'
        }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${checked ? 'translate-x-6' : 'translate-x-1'
          }`}
      />
    </button>
  );
}

export default function ChatTitle({ title, setTitle, showRedaction = false, onToggleRedaction }: ChatTitleProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState(title);
  
  useEffect(() => {
    setInputValue(title);
  }, [title]);
  
  return (
    <div className="flex items-center justify-between w-full px-4">
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
              <div className="flex gap-2">
                <SaveButton onClick={() => {
                  setIsEditing(false);
                  setTitle(inputValue);
                }} />
                <CancelButton onClick={() => {
                  setIsEditing(false);
                  setInputValue(title);
                }} />
              </div>
            </>
          ) : (
            <>
              <span
                className="text-xl font-medium"
                onClick={() => setIsEditing(true)}
              >
                {title}
              </span>
              <EditButton onClick={() => setIsEditing(true)} />
            </>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-600">Show redaction</span>
        <Toggle checked={showRedaction} onChange={onToggleRedaction} />
      </div>
    </div>
  );
}