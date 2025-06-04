import React from 'react';
import { Plus, Trash } from 'lucide-react';
import { NEW_CHAT_ID, ChatPreview } from '@/hooks/useSafeGPT';

interface SidebarProps {
  items: ChatPreview[];
  selectedId?: string;
  padding?: number;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function Sidebar({ items, onSelect, selectedId, padding, onDelete }: SidebarProps) {
  return (
    <div className="w-full h-full overflow-y-auto bg-[rgb(252,252,249)] border-r border-gray-200">
      <ul className="flex flex-col mt-8">
        {items.map((item) => (
          <li key={item.id}>
            <button
              className={`flex w-[96%] text-left py-2 transition-colors duration-150 cursor-pointer focus:outline-none items-center justify-between rounded-xl mx-[2%]
                ${selectedId === item.id ? 'bg-[rgb(85,152,229)]/10 text-[rgb(85,152,229)] font-semibold' : 'hover:bg-[rgb(85,152,229)]/5'}
              `}
              style={{ paddingLeft: padding || 16, paddingRight: 16 }}
              onClick={() => onSelect(item.id)}
            >
              {item.title}
              {item.id !== NEW_CHAT_ID && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(item.id);
                  }}
                  className="hover:text-red-500 transition-colors duration-200"
                >
                  <Trash size={18} />
                </button>
              )}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
