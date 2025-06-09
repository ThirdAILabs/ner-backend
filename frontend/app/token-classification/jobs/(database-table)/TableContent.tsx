import React, { useMemo } from 'react';
import { TableHead, TableRow, TableHeader, TableBody, TableCell } from '@/components/ui/table';
import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { NO_GROUP } from '@/lib/utils';
import { ChevronRight } from 'lucide-react';
import { TokenHighlighter } from '@/components/feedback/TokenHighlighter';
import * as _ from 'lodash';

const PASTELS = ['#E5A49C', '#F6C886', '#FBE7AA', '#99E3B5', '#A6E6E7', '#A5A1E1', '#D8A4E2'];
const DARKERS = ['#D34F3E', '#F09336', '#F7CF5F', '#5CC96E', '#65CFD0', '#597CE2', '#B64DC8'];

const DEFAULT_COLOR = { text: '#E0E0E0', tag: '#A0A0A0' };

interface HighlightColor {
  text: string;
  tag: string;
}

interface HighlightedTokenProps {
  token: string;
  tag: string;
  tagColors: Record<string, HighlightColor>;
  labeled: boolean;
}

const HighlightedToken = React.memo(({ token, tag, tagColors, labeled }: HighlightedTokenProps) => {
  const needsSpaceBefore = !(token.match(/^[.,;:!?)\]}"'%]/) || token.trim() === '');

  const needsSpaceAfter = !(
    token.match(/^[([{"'$]/) ||
    token.match(/[.,;:!?]$/) ||
    token.trim() === ''
  );

  if (tag === 'O') {
    return (
      <span>
        {needsSpaceBefore && token !== '' && ' '}
        {token}
      </span>
    );
  }

  const tagColor = tagColors[tag] || DEFAULT_COLOR;

  return (
    <span>
      {needsSpaceBefore && ' '}
      <span
        style={{
          backgroundColor: tagColor.text,
          padding: '2px 4px',
          borderRadius: '2px',
          userSelect: 'none',
          display: 'inline-flex',
          alignItems: 'center',
          wordBreak: 'break-word',
        }}
      >
        {token}
        {labeled && (
          <span
            style={{
              backgroundColor: tagColor.tag,
              color: 'white',
              fontSize: '11px',
              fontWeight: 'bold',
              borderRadius: '2px',
              marginLeft: '4px',
              padding: '1px 3px',
            }}
          >
            {tag}
          </span>
        )}
      </span>
    </span>
  );
});

const HighlightedTag = React.memo(
  ({ tag, tagColors }: { tag: string; tagColors: Record<string, HighlightColor> }) => {
    if (tag === 'O') {
      return <span>{tag}</span>;
    }

    const tagColor = tagColors[tag] || DEFAULT_COLOR;

    return (
      <span
        style={{
          backgroundColor: tagColor.tag,
          color: 'white',
          fontSize: '11px',
          fontWeight: 'bold',
          borderRadius: '2px',
          padding: '1px 3px',
          display: 'inline-block',
        }}
      >
        {tag}
      </span>
    );
  }
);

interface TokenContextProps {
  context?: { left: string; right: string; token: string; tag: string };
  tagColors: Record<string, HighlightColor>;
}

const TokenContext = React.memo(({ context, tagColors }: TokenContextProps) => {
  if (!context) return <span className="text-red-400 text-xs">No context available</span>;

  const leftContent = context.left || '';
  const rightContent = context.right || '';
  const token = context.token || '';
  const tag = context.tag || '';

  return (
    <div className="font-mono text-xs border border-gray-200 p-2 rounded bg-gray-50">
      <span className="text-gray-600">{leftContent}</span>
      <HighlightedToken token={token} tag={tag} tagColors={tagColors} labeled />
      <span className="text-gray-600">{rightContent}</span>
    </div>
  );
});

const LoadMoreButton = React.memo(
  ({
    hasMore,
    isLoading,
    onClick,
  }: {
    hasMore: boolean;
    isLoading: boolean;
    onClick: () => void;
  }) => {
    if (!hasMore) return null;

    return (
      <div className="py-4 flex justify-center">
        <Button
          variant="outline"
          onClick={onClick}
          disabled={isLoading || !hasMore}
          className="w-full max-w-sm"
        >
          {isLoading ? (
            <div className="flex items-center justify-center">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Loading more...
            </div>
          ) : (
            'Load More'
          )}
        </Button>
      </div>
    );
  }
);

export function TableContent({
  viewMode,
  objectRecords,
  tokenRecords,
  groupFilters,
  tagFilters,
  isLoadingObjectRecords,
  isLoadingTokenRecords,
  tags,
  hasMoreTokens = false,
  hasMoreObjects = false,
  onLoadMore,
  showFilterContent,
  pathMap,
  addFeedback,
}: TableContentProps) {
  const tagColors = useMemo(() => {
    const colors: Record<string, HighlightColor> = {};
    tags
      .filter((tag) => tag.type !== 'O')
      .forEach((tag, index) => {
        colors[tag.type] = {
          text: PASTELS[index % PASTELS.length],
          tag: DARKERS[index % DARKERS.length],
        };
      });
    return colors;
  }, [tags]);

  const filterRecords = (recordGroups: string[], recordTags: string[]) => {
    const matchTags = recordTags.some((tag) => tagFilters[tag] !== false);
    const matchNoGroup = recordGroups.length === 0 && groupFilters[NO_GROUP];
    const matchUserDefinedGroup = recordGroups.some((group) => groupFilters[group] !== false);
    const noGroupConfigured = Object.keys(groupFilters).length === 0;
    return matchTags && (matchNoGroup || matchUserDefinedGroup || noGroupConfigured);
  };

  const handleFullPath = (fileIdentifier: string) => {
    const fullPath = pathMap?.[fileIdentifier.split('/').slice(-1).join('')];
    const openFile = () => {
      // @ts-ignore
      window.electron?.openFile?.(fullPath);
    };
    return { fullPath, openFile };
  };

  const truncateFilePath = (filePath: string) => {
    const maxLength = 50;
    if (filePath.length > maxLength) {
      return '...' + filePath.slice(filePath.length - maxLength, filePath.length);
    }
    return filePath;
  };

  if (viewMode === 'classified-token') {
    const filteredRecords = tokenRecords.filter((record) =>
      filterRecords(record.groups, [record.tag])
    );

    return (
      <>
        <TableHeader>
          <TableRow>
            <TableHead>Prediction</TableHead>
            <TableHead>Source Object</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filteredRecords.length > 0 ? (
            filteredRecords.map((record, index) => {
              let tokens = [
                [record.context?.left || '', 'O'],
                [record.token, record.tag],
                [record.context?.right || '', 'O'],
              ].flatMap((token) =>
                token[0]
                  .split(/\s+/)
                  .filter((word) => word.trim() !== '')
                  .map((word) => ({ text: word, tag: token[1] }))
              );
              return (
                <TableRow key={index}>
                  <TableCell className="w-3/5">
                    <TokenHighlighter tokens={tokens} availableTags={tags.map((tag) => tag.type)} />
                  </TableCell>
                  <TableCell className="w-1/5 px-4">
                    <div className="relative group">
                      {(() => {
                        const fileIdentifier = record.sourceObject;
                        const { fullPath, openFile } = handleFullPath(fileIdentifier);
                        // @ts-ignore
                        if (fullPath && typeof window !== 'undefined' && window.electron) {
                          return (
                            <span
                              style={{
                                textDecoration: 'underline',
                                color: 'inherit',
                                cursor: 'pointer',
                              }}
                              title={fileIdentifier.split('/').slice(-1).join('')}
                              onClick={openFile}
                            >
                              {truncateFilePath(fullPath)}
                            </span>
                          );
                        } else {
                          return (
                            <span
                              style={{ color: 'inherit' }}
                              title={fileIdentifier.split('/').slice(-1).join('')}
                            >
                              {fileIdentifier.split('/').slice(-1)}
                            </span>
                          );
                        }
                      })()}
                    </div>
                  </TableCell>
                </TableRow>
              );
            })
          ) : (
            <TableRow>
              <TableCell colSpan={2} className="text-center py-8 text-gray-500">
                {tokenRecords.length === 0 ? (
                  isLoadingTokenRecords ? (
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading records...
                    </div>
                  ) : (
                    <div>No token records found. Check API response.</div>
                  )
                ) : (
                  <div>No records match the current filters.</div>
                )}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </>
    );
  }

  const filteredRecords = objectRecords.filter((record) =>
    filterRecords(record.groups, [
      ...new Set(record.taggedTokens.map((token) => token[1]).filter((tag) => tag !== 'O')),
    ])
  );

  return (
    <div className="mt-4">
      {filteredRecords.length === 0 ? (
        <div className="text-gray-500">No records match the current filters.</div>
      ) : (
        filteredRecords.map((record, index) => {
          const fileIdentifier = record.sourceObject;
          const { fullPath, openFile } = handleFullPath(fileIdentifier);
          const tokens = record.taggedTokens.flatMap((token) => {
            const [text, tag] = token;
            return text
              .split(/\s+/)
              .filter((word) => word.trim() !== '')
              .map((word) => ({ text: word, tag }));
          });

          const onTagAssign = (startIndex: number, endIndex: number, newTag: string) => {
            const leftContext = tokens
              .slice(Math.max(0, startIndex - 5), startIndex)
              .map((t) => t.text)
              .join(' ');
            const highlightedText = tokens
              .slice(startIndex, endIndex + 1)
              .map((t) => t.text)
              .join(' ');
            const rightContext = tokens
              .slice(endIndex + 1, Math.min(endIndex + 6, tokens.length))
              .map((t) => t.text)
              .join(' ');
            addFeedback(
              {
                highlightedText,
                tag: newTag,
                leftContext,
                rightContext,
                startIndex,
                endIndex,
                objectId: fileIdentifier,
              },
              tokens.map((token) => token.text),
              tokens.map((token) => token.tag)
            );
          };

          return (
            <details
              key={index}
              className="group text-sm leading-relaxed bg-white rounded border border-gray-100 shadow-sm mb-4"
            >
              <summary className="p-3 cursor-pointer bg-gray-100 flex items-center">
                <div className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 transition-transform group-open:rotate-90" />
                  {/* @ts-ignore */}
                  {fullPath && typeof window !== 'undefined' && window.electron ? (
                    <span
                      className="font-semibold"
                      style={{
                        textDecoration: 'underline',
                        color: 'inherit',
                        cursor: 'pointer',
                        userSelect: 'none',
                      }}
                      onClick={openFile}
                    >
                      {truncateFilePath(fullPath)}
                    </span>
                  ) : (
                    <span
                      className="font-semibold"
                      style={{ color: 'inherit', userSelect: 'none' }}
                    >
                      {fileIdentifier.split('/').slice(-1)}
                    </span>
                  )}
                </div>
              </summary>
              <div className="p-4">
                <TokenHighlighter
                  tokens={tokens}
                  editable
                  availableTags={tags.map((tag) => tag.type)}
                  onTagAssign={onTagAssign}
                />
                ...
                <p className="text-gray-500 text-xs">
                  Truncated File View. Please open the original file for the entire content.
                </p>
              </div>
            </details>
          );
        })
      )}
    </div>
  );
}
