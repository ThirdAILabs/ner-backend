import { Menu, MenuItem } from '@mui/material';
import React, { useRef, useState } from 'react';
import { Tooltip } from './Tooltip';
import { environment } from '@/lib/environment';

const REMOVE_TAG_NAME = 'Remove Tag';

const InfoIcon = () => {
  return (
    <span className="flex items-center ml-2 text-xs text-gray-500">
      <svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24" className="cursor-help">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
      </svg>
    </span>
  );
};

interface TagDropdownProps {
  position: { x: number; y: number } | null;
  tagColors: Record<string, string>;
  unassignableTags: string[];
  showRemoveTag: boolean;
  onClose: () => void;
  onTagSelect: (tag: string) => void;
}

export const TagDropdown: React.FC<TagDropdownProps> = ({
  position,
  tagColors,
  unassignableTags,
  showRemoveTag,
  onClose,
  onTagSelect,
}) => {
  const [query, setQuery] = useState('');
  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null);
  const [justOpenedDropdown, setJustOpenedDropdown] = useState(true);
  const dropdownQueryRef = useRef<HTMLInputElement>(null);

  const availableTags = Object.keys(tagColors).sort();
  const queryMatchesTag =
    query && availableTags.some((tag) => tag.toLowerCase() === query.toLowerCase());
  const filteredTags = query
    ? availableTags.filter((tag) => tag.toLowerCase().includes(query.toLowerCase()))
    : availableTags;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
  };

  const relayTypingToInputField = (e: React.KeyboardEvent<HTMLDivElement>) => {
    // Keystrokes are relayed by the dropdown menu to the input field inside of it
    // except for navigation keys. This creates a smoother user experience.
    if (
      e.key === 'ArrowUp' ||
      e.key === 'ArrowDown' ||
      e.key === 'ArrowLeft' ||
      e.key === 'ArrowRight' ||
      e.key === 'Enter'
    ) {
      return;
    }
    dropdownQueryRef.current?.focus();
  };

  const handleInitialOpen = () => {
    // Immediately shows caret in the input field and makes up and down arrows
    // work immediately upon opening the dropdown.
    if (justOpenedDropdown) {
      setJustOpenedDropdown(false);
      dropdownQueryRef.current?.focus();
    }
  };

  const handleClose = () => {
    onClose();
    setQuery('');
    // Reset justOpenedDropdown for the next time the menu is opened.
    setJustOpenedDropdown(true);
  };

  const stopPropagationExceptNavigation = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Except for navigation keys, don't propagate the keystroke to the menu component
    // so the selector does not jump around, creating a smooth user experience.
    if (
      e.key === 'ArrowUp' ||
      e.key === 'ArrowDown' ||
      e.key === 'ArrowLeft' ||
      e.key === 'ArrowRight'
    ) {
      return;
    }
    e.stopPropagation();
  };

  const displayUnsassignableTagTooltip = (tag: string, e: React.MouseEvent<HTMLElement>) => {
    setTooltip({
      text: `${tag} cannot be selected because it is a custom tag`,
      x: e.clientX,
      y: e.clientY,
    });
  };

  const hideUnsassignableTagTooltip = () => {
    setTooltip(null);
  };

  const displayNoMatchesFound = () => {
    if (environment.allowNewTagsInFeedback)
      return (
        <MenuItem
          onClick={() => onTagSelect(query)}
          className="flex items-center justify-between font-bold text-black bg-white rounded-md mx-2 my-0.5 px-3 py-2 transition-colors hover:bg-gray-100"
        >
          <span>{query}</span>
          <span className="ml-3 bg-gray-200 text-gray-700 rounded-full text-xs font-medium px-3 py-0.5">
            new
          </span>
        </MenuItem>
      );

    return <div className="text-gray-500 text-sm px-4 pb-2">No matches found</div>;
  };

  return (
    <>
      <Menu
        open={position !== null}
        anchorReference="anchorPosition"
        anchorPosition={position ? { top: position.y, left: position.x } : undefined}
        onKeyDown={relayTypingToInputField}
        onFocus={handleInitialOpen}
        onClose={handleClose}
        PaperProps={{
          sx: {
            borderRadius: '8px',
            overflow: 'hidden',
          },
        }}
      >
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onKeyDown={stopPropagationExceptNavigation}
          className="w-full p-2 border border-gray-300 rounded mb-1 bg-white text-base"
          placeholder="Search Tags..."
          autoFocus
          ref={dropdownQueryRef}
          style={{
            padding: '10px 15px',
            border: 'none',
            outline: 'none',
            boxShadow: 'none',
            backgroundColor: 'transparent',
          }}
        />

        {query && !queryMatchesTag && displayNoMatchesFound()}

        {showRemoveTag && (
          <MenuItem
            key={REMOVE_TAG_NAME}
            value={REMOVE_TAG_NAME}
            onClick={() => {
              onTagSelect('O');
            }}
          >
            <svg
              width="16"
              height="16"
              fill="none"
              stroke="#ef4444"
              strokeWidth="2"
              viewBox="0 0 24 24"
              style={{
                marginRight: 8,
                display: 'inline-block',
                verticalAlign: 'middle',
              }}
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
            <span>Remove Tag</span>
          </MenuItem>
        )}

        {filteredTags.map((tag) => (
          <MenuItem
            key={tag}
            value={tag}
            onClick={(e) => {
              // This effectively disables this menu item.
              // We cannot just pass a disabled=True prop because the tooltip wouldn't work.
              if (unassignableTags.includes(tag)) {
                e.stopPropagation();
                return;
              }
              onTagSelect(tag);
            }}
            onMouseOver={(e) => {
              if (unassignableTags.includes(tag)) {
                displayUnsassignableTagTooltip(tag, e);
              }
            }}
            onMouseLeave={() => {
              if (unassignableTags.includes(tag)) {
                hideUnsassignableTagTooltip();
              }
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center' }}>
              <span
                style={{
                  display: 'inline-block',
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  background: tagColors[tag]!,
                  marginRight: 8,
                  verticalAlign: 'middle',
                }}
              />
              <span style={{ color: unassignableTags.includes(tag) ? 'grey' : 'black' }}>
                {tag}
              </span>
            </span>

            {unassignableTags.includes(tag) && <InfoIcon />}
          </MenuItem>
        ))}
      </Menu>

      {tooltip && <Tooltip {...tooltip} />}
    </>
  );
};
