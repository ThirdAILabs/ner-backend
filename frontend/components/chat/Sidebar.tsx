import React from 'react';
import { Plus } from 'lucide-react';

export interface ChatPreview {
  id: string;
  title: string;
}

interface SidebarProps {
  items: ChatPreview[];
  selectedId?: string;
  padding?: number;
  onSelect: (id: string) => void;
  onNewChat: () => void;
}

function NewChatButton({ onNewChat }: { onNewChat: () => void }) {
  return (
    <button
      onClick={onNewChat}
      className="w-full flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-[rgb(85,152,229)]/5 transition-colors duration-200 border border-gray-200 group"
      aria-label="New chat"
    >
      <div className="w-8 h-8 rounded-full bg-[rgb(85,152,229)] flex items-center justify-center text-white shadow-sm group-hover:bg-[rgb(85,152,229)]/90 group-hover:scale-105 transition-all duration-200">
        <Plus size={18} />
      </div>
      <span className="text-gray-700">New chat</span>
    </button>
  );
}

export default function Sidebar({ items, onSelect, selectedId, padding, onNewChat }: SidebarProps) {
  return (
    <div className="w-full h-full overflow-y-auto bg-white border-r border-gray-200">
      <div className="p-4">
        <NewChatButton onNewChat={onNewChat} />
      </div>
      <ul className="flex flex-col">
        {items.map((item) => (
          <li key={item.id}>
            <button
              className={`w-full text-left py-3 transition-colors duration-150 cursor-pointer focus:outline-none
                ${selectedId === item.id ? 'bg-[rgb(85,152,229)]/10 text-[rgb(85,152,229)] font-semibold' : 'hover:bg-[rgb(85,152,229)]/5'}
              `}
              style={{ paddingLeft: padding || 16, paddingRight: 16 }}
              onClick={() => onSelect(item.id)}
            >
              {item.title}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
