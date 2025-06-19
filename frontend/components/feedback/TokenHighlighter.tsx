/**
 * A text box where tokens can be highlighted and assigned a tag.
 */

import React, { useState, useRef, useMemo } from 'react';
import { DisplayedToken } from '@/components/feedback/useFeedbackState';
import { FeedbackMetadata } from '@/components/feedback/useFeedbackState';
import { TagDropdown } from './TagDropdown';

interface HighlightColor {
  text: string;
  tag: string;
}

interface TokenHighlighterProps {
  tokens: DisplayedToken[];
  availableTags: string[];
  unassignableTags?: string[];
  spotlightStartIndex?: number;
  spotlightEndIndex?: number;
  editable?: boolean;
  onTagAssign?: (startIndex: number, endIndex: number, newTag: string) => void;
  objectId?: string;
  tagFilters?: Record<string, boolean>;
}

interface Feedback {
  id: string;
  body: FeedbackMetadata;
}

const HOVERING_COLOR = '#EFEFEF';
const SELECTING_COLOR = '#EFEFEF';
const NEUTRAL_TAG_COLOR = '#666';
const PASTELS = ['#E5A49C', '#F6C886', '#FBE7AA', '#99E3B5', '#A6E6E7', '#A5A1E1', '#D8A4E2'];
const DARKERS = ['#D34F3E', '#F09336', '#F7CF5F', '#5CC96E', '#65CFD0', '#597CE2', '#B64DC8'];
const NEW_TAG_COLOR = {
  text: '#F1F5F9',
  tag: '#64748B',
};

export const TokenHighlighter: React.FC<TokenHighlighterProps> = ({
  tokens,
  availableTags,
  unassignableTags = [],
  spotlightStartIndex,
  spotlightEndIndex,
  editable = false,
  onTagAssign,
  objectId,
  tagFilters,
}) => {
  const [selectionStart, setSelectionStart] = useState<number | null>(null);
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [dropdownPosition, setDropdownPosition] = useState<{
    x: number;
    y: number;
  } | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const nonTrivialTagColors = useMemo(() => {
    const colors: Record<string, HighlightColor> = {};

    availableTags
      .filter((tag) => tag !== 'O')
      .forEach((tag, index) => {
        colors[tag] = {
          text: PASTELS[index % PASTELS.length],
          tag: DARKERS[index % DARKERS.length],
        };
      });

    tokens.forEach((token) => {
      if (token.tag !== 'O' && !colors[token.tag]) {
        colors[token.tag] = NEW_TAG_COLOR;
      }
    });

    return colors;
  }, [availableTags, tokens]);

  const handleMouseDown = (index: number) => {
    if (!editable) return;
    setSelectionStart(index);
    setSelectionEnd(index);
    setIsSelecting(true);
  };

  const handleMouseOver = (index: number) => {
    if (!editable) return;
    setHoveredIndex(index);
    if (selectionStart !== null) {
      setSelectionEnd(index);
    }
  };

  const handleMouseLeaveToken = (e: React.MouseEvent) => {
    if (!editable) return;
    setHoveredIndex(null);
    e.stopPropagation();
  };

  const handleMouseLeaveFeedbackBox = () => {
    if (!editable) return;
    setIsSelecting(false);
    setSelectionStart(null);
    setSelectionEnd(null);
  };

  const handleMouseUp = (event: React.MouseEvent) => {
    if (!editable) return;
    if (isSelecting) {
      setDropdownPosition({
        x: event.clientX,
        y: event.clientY,
      });
      setIsSelecting(false);
    }
  };

  const handleTagSelect = (tag: string) => {
    if (!editable) return;
    if (selectionStart === null || selectionEnd === null) return;
    const start = Math.min(selectionStart, selectionEnd);
    const end = Math.max(selectionStart, selectionEnd);
    if (onTagAssign) {
      onTagAssign(start, end, tag);
    }
    setDropdownPosition(null);
    setSelectionStart(null);
    setSelectionEnd(null);
  };

  // Display the 'Remove Tag' option only if there are tags to remove;
  // it doesn't make sense to remove tags when the whole span consists of "O" tags.
  const canRemoveTagFromSelection =
    selectionStart !== null &&
    selectionEnd !== null &&
    tokens
      .slice(Math.min(selectionStart, selectionEnd), Math.max(selectionStart, selectionEnd) + 1)
      .some((token) => token.tag !== 'O' && !unassignableTags.includes(token.tag));

  const getReportId = () => {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('jobId') || '';
  };

  const reportId = getReportId();
  const FEEDBACK_STORAGE_KEY = `feedback-${reportId}`;
  const initialFeedback: Feedback[] = JSON.parse(
    localStorage.getItem(FEEDBACK_STORAGE_KEY) || '[]'
  );

  function isFeedbackGiven(tokenIndex: number, objectId: string) {
    for (let index = 0; index < initialFeedback.length; index++) {
      const feedback = initialFeedback[index];

      const isTokenMatched =
        feedback.body.objectId === objectId &&
        feedback.body.startIndex <= tokenIndex &&
        feedback.body.endIndex >= tokenIndex;

      if (isTokenMatched) {
        return true;
      }
    }
    return false;
  }

  return (
    <div ref={containerRef} onMouseUp={handleMouseUp} onMouseLeave={handleMouseLeaveFeedbackBox}>
      <div className="flex flex-wrap gap-0.5">
        {tokens.map((token, index) => {
          const selectionMade = selectionStart !== null && selectionEnd !== null;
          const startIndex = selectionMade ? Math.min(selectionStart, selectionEnd) : tokens.length;
          const endIndex = selectionMade ? Math.max(selectionStart, selectionEnd) : -1;
          const isSelected = index >= startIndex && index <= endIndex;
          const isSpotlighted =
            spotlightStartIndex &&
            spotlightEndIndex &&
            index >= spotlightStartIndex &&
            index <= spotlightEndIndex;

          // Explicitly check if tagFilters[token.tag] is false instead of using !tagFilters[token.tag]
          // because tagFilters[token.tag] could be undefined for trivial and new tags.
          const isFilteredOut = tagFilters && tagFilters[token.tag] === false;

          // Default values for trivial tags.
          let textBackgroundColor = 'transparent';
          let textBorder = 'none';
          let showTag = false;
          let tagBackgroundColor = NEUTRAL_TAG_COLOR;
          let tagTextColor = 'white';

          // This is equivalent to token.tag !== 'O' && nonTrivialTagColors[token.tag]
          if (nonTrivialTagColors[token.tag]) {
            textBackgroundColor = nonTrivialTagColors[token.tag].text;
            tagBackgroundColor = nonTrivialTagColors[token.tag].tag;
            showTag = true;
          }

          if (isSpotlighted) {
            textBorder = `1px dotted ${tagBackgroundColor}`;
          }

          if (isFilteredOut) {
            textBorder = `1px solid ${tagBackgroundColor}`;
            tagTextColor = tagBackgroundColor;
            textBackgroundColor = 'transparent';
            tagBackgroundColor = 'transparent';
          }

          if (isSelected) {
            textBackgroundColor = SELECTING_COLOR;
          }

          if (hoveredIndex === index && editable) {
            textBackgroundColor = HOVERING_COLOR;
          }

          return (
            <span
              key={index}
              onMouseOver={() => handleMouseOver(index)}
              onMouseLeave={handleMouseLeaveToken}
              className={[
                `inline-flex items-center ${editable ? 'cursor-pointer' : 'cursor-default'} select-none rounded-sm m-0 py-0.5 ${token.tag === 'O' ? 'px-0.5' : 'px-1'}`,
              ].join(' ')}
              style={{
                backgroundColor: textBackgroundColor,
                color: 'black',
                border: textBorder,
              }}
              onMouseDown={() => handleMouseDown(index)}
            >
              <span
                style={{
                  textDecoration:
                    objectId && isFeedbackGiven(index, objectId) ? 'underline' : 'none',
                  textDecorationColor:
                    objectId && isFeedbackGiven(index, objectId) ? '#0000EE' : 'none',
                  textDecorationThickness: '1px',
                }}
              >
                {token.text}
              </span>

              {showTag && (index === tokens.length - 1 || tokens[index + 1]?.tag !== token.tag) && (
                <span
                  className="ml-1 rounded-sm px-1.5 py-0.5 text-xs font-bold"
                  style={{
                    backgroundColor: tagBackgroundColor,
                    color: tagTextColor,
                    textDecoration: 'none',
                  }}
                >
                  {token.tag}
                </span>
              )}
            </span>
          );
        })}
      </div>

      <TagDropdown
        position={dropdownPosition}
        tagColors={Object.fromEntries(
          Object.entries(nonTrivialTagColors).map(([tag, color]) => [tag, color.text])
        )}
        unassignableTags={unassignableTags}
        showRemoveTag={canRemoveTagFromSelection}
        onClose={() => {
          setDropdownPosition(null);
          setSelectionStart(null);
          setSelectionEnd(null);
        }}
        onTagSelect={handleTagSelect}
      />
    </div>
  );
};
