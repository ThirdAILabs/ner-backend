import React from 'react';

export interface ChatPreview {
  id: string;
  title: string;
}

interface SidebarProps {
  items: ChatPreview[];
  onSelect: (id: string) => void;
  selectedId?: string;
  padding?: number;
}

export default function Sidebar({ items, onSelect, selectedId, padding }: SidebarProps) {
  return (
    <div className="w-full h-full overflow-y-auto bg-white border-r border-gray-200">
      <ul className="flex flex-col">
        {items.map((item) => (
          <li key={item.id}>
            <button
              className={`w-full text-left py-3 transition-colors duration-150 cursor-pointer focus:outline-none
                ${selectedId === item.id ? 'bg-[rgb(85,152,229)]/10 text-[rgb(85,152,229)] font-semibold' : 'hover:bg-[rgb(85,152,229)]/5'}
              `}
              style={{paddingLeft: padding || 16, paddingRight: 16}}
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
