import React, { useState, useRef, useEffect } from 'react';
import { TokenHighlighter } from './TokenHighlighter';
import { Trash2 } from 'lucide-react';

interface Token {
  text: string;
  tag: string;
}

interface FeedbackPanelProps {
  feedbacks: {
    id: string;
    tokens: Token[];
    spotlightStartIndex: number;
    spotlightEndIndex: number;
  }[];
  availableTags: string[];
  onDelete: (id: string) => void;
  onSubmit: () => void;
}

interface FeedbackRowProps {
  id: string;
  tokens: Token[];
  availableTags: string[];
  spotlightStartIndex: number;
  spotlightEndIndex: number;
  onDelete: (id: string) => void;
}

const FeedbackRow: React.FC<FeedbackRowProps> = ({
  id,
  tokens,
  availableTags,
  spotlightStartIndex,
  spotlightEndIndex,
  onDelete,
}) => {
  const [confirm, setConfirm] = useState(false);
  const confirmBtnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!confirm) return;
    function handleClick(e: MouseEvent) {
      if (confirmBtnRef.current && !confirmBtnRef.current.contains(e.target as Node)) {
        setConfirm(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [confirm]);

  return (
    <div className="flex items-start justify-between w-full gap-2 py-2">
      <div className="flex-grow min-w-0">
        <TokenHighlighter
          tokens={tokens}
          availableTags={availableTags}
          spotlightStartIndex={spotlightStartIndex}
          spotlightEndIndex={spotlightEndIndex}
        />
      </div>
      <div className="flex-shrink-0 ml-2 flex mt-1">
        {confirm ? (
          <button
            ref={confirmBtnRef}
            className="bg-red-500 text-white rounded-full px-3 py-1 text-xs font-semibold shadow hover:bg-red-600 transition"
            onClick={() => {
              onDelete(id);
              setConfirm(false);
            }}
          >
            Confirm
          </button>
        ) : (
          <button
            className="p-0 m-0 bg-transparent border-none outline-none flex items-center justify-center text-red-500 hover:text-red-700 transition-colors duration-200"
            onClick={() => setConfirm(true)}
            aria-label="Delete feedback"
          >
            <Trash2 size={18} />
          </button>
        )}
      </div>
    </div>
  );
};

export const FeedbackPanel: React.FC<FeedbackPanelProps> = ({
  feedbacks,
  availableTags,
  onDelete,
  onSubmit,
}) => {
  const [collapsed, setCollapsed] = useState(feedbacks.length === 0);
  const prevFeedbackCountRef = useRef(feedbacks.length);

  useEffect(() => {
    if (prevFeedbackCountRef.current === 0 && feedbacks.length > 0) {
      setCollapsed(false);
    }
    prevFeedbackCountRef.current = feedbacks.length;
  }, [feedbacks]);

  return (
    <div
      className={
        collapsed
          ? 'bg-blue-600 rounded-lg shadow-lg border border-gray-200 w-full h-12 flex items-center justify-between px-4 cursor-pointer transition-all duration-200 hover:bg-blue-700 '
          : 'flex flex-col h-full w-full bg-white rounded-lg shadow-lg border border-gray-200 transition-all duration-200'
      }
      onClick={collapsed ? () => setCollapsed(false) : undefined}
    >
      {/* Collapsed bar */}
      {collapsed ? (
        <>
          <span className="text-base font-semibold text-white">Feedback ({feedbacks.length})</span>
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
        </>
      ) : (
        <>
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 rounded-t-lg bg-white flex-shrink-0">
            <span className="text-base font-semibold text-gray-800">
              Feedback ({feedbacks.length})
            </span>
            <button
              className="p-1 rounded hover:bg-gray-100 transition"
              onClick={() => setCollapsed((c) => !c)}
              aria-label={collapsed ? 'Expand' : 'Collapse'}
              type="button"
            >
              {/* Minus icon when expanded */}
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="5" y1="12" x2="19" y2="12"></line>
              </svg>
            </button>
          </div>

          {/* Feedback list */}
          <div className="flex-1 overflow-y-auto px-4 py-1 divide-y divide-gray-200">
            {feedbacks.map((fb) => (
              <FeedbackRow
                key={fb.id}
                id={fb.id}
                tokens={fb.tokens}
                availableTags={availableTags}
                spotlightStartIndex={fb.spotlightStartIndex}
                spotlightEndIndex={fb.spotlightEndIndex}
                onDelete={onDelete}
              />
            ))}
          </div>

          {/* Submit button */}
          <div className="px-4 pb-4 flex-shrink-0">
            <button
              className="w-full bg-blue-600 text-white border border-blue-700 rounded-md py-2 text-base font-medium hover:bg-blue-700 transition shadow-md"
              onClick={onSubmit}
              style={{ position: 'relative', zIndex: 10 }}
              type="button"
            >
              Submit
            </button>
          </div>
        </>
      )}
    </div>
  );
};
