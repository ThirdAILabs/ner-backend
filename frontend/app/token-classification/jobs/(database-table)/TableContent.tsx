import React, { useMemo } from 'react';
import {
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
} from '@/components/ui/table';
import { Loader2 } from 'lucide-react';
import { ClassifiedTokenDatabaseRecord, ObjectDatabaseRecord, ViewMode, TableContentProps } from './types';
import { Button } from '@/components/ui/button';

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
}

const HighlightedToken = React.memo(({ token, tag, tagColors }: HighlightedTokenProps) => {
  const needsSpaceBefore = !(
    token.match(/^[.,;:!?)\]}"'%]/) || 
    token.trim() === ''
  );
  
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
      </span>
    </span>
  );
});

const HighlightedTag = React.memo(({ tag, tagColors }: { tag: string; tagColors: Record<string, HighlightColor> }) => {
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
});

interface TokenContextProps {
  context?: { left: string; right: string; token: string; tag: string };
  tagColors: Record<string, HighlightColor>;
}

const TokenContext = React.memo(({ context, tagColors }: TokenContextProps) => {
  if (!context) return <span className="text-red-400 text-xs">No context available</span>;

  const leftContent = context.left || '[empty left context]';
  const rightContent = context.right || '[empty right context]';
  const token = context.token || '[empty token]';
  const tag = context.tag || '[empty tag]';

  return (
    <div className="font-mono text-xs border border-gray-200 p-2 rounded bg-gray-50">
      <span className="text-gray-600">{leftContent}</span>
      <HighlightedToken token={token} tag={tag} tagColors={tagColors} />
      <span className="text-gray-600">{rightContent}</span>
    </div>
  );
});

const LoadMoreButton = React.memo(({ hasMore, isLoading, onClick }: { hasMore: boolean, isLoading: boolean, onClick: () => void }) => {
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
});

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
}: TableContentProps) {

  const tagColors = useMemo(() => {
    const colors: Record<string, HighlightColor> = {};
    tags
      .filter(tag => tag.type !== 'O')
      .forEach((tag, index) => {
        colors[tag.type] = {
          text: PASTELS[index % PASTELS.length],
          tag: DARKERS[index % DARKERS.length],
        };
      });
    return colors;
  }, [tags]);

  if (viewMode === 'classified-token') {
    const filteredRecords = tokenRecords.filter((record) => {
      const tagMatches = tagFilters[record.tag] !== false;
      const groupMatches = record.groups.length === 0 || record.groups.some((group) => groupFilters[group] !== false);
      return tagMatches && groupMatches;
    });

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
            filteredRecords.map((record, index) => (
              <TableRow key={index}>
                <TableCell>
                  {record.context ? (
                    <TokenContext context={{
                      left: record.context.left,
                      right: record.context.right,
                      token: record.token,
                      tag: record.tag,
                    }} tagColors={tagColors} />
                  ) : (
                    <span className="text-red-400 text-xs">Missing context</span>
                  )}
                </TableCell>
                <TableCell>{record.sourceObject}</TableCell>
              </TableRow>
            ))
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

          {filteredRecords.length > 0 && (
            <TableRow>
              <TableCell colSpan={2}>
                <LoadMoreButton hasMore={hasMoreTokens} isLoading={isLoadingTokenRecords} onClick={onLoadMore ?? (() => {})} />

              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </>
    );
  }

  if (viewMode === 'object') {
    const filteredRecords = objectRecords.filter((record) => {
      return record.groups.length === 0 || record.groups.some((group) => groupFilters[group]);
    });

    return (
      <>
        <TableHeader>
          <TableRow>
            <TableHead>Full Text with Tagged Tokens</TableHead>
            <TableHead>Source Object</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {filteredRecords.length > 0 ? (
            filteredRecords.map((record, index) => (
              <TableRow key={index}>
                <TableCell
                  style={{
                    maxWidth: '60%',
                    whiteSpace: 'normal',
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word',
                    padding: '16px',
                  }}
                >
                  <div className="text-sm leading-relaxed bg-white p-3 rounded border border-gray-100 shadow-sm">
                    {record.taggedTokens.map((token, tokenIndex) => (
                      <HighlightedToken
                        key={`${index}-${tokenIndex}`}
                        token={token[0]}
                        tag={tagFilters[token[1]] ? token[1] : "O"}
                        tagColors={tagColors}
                      />
                    ))}
                  </div>
                </TableCell>
                <TableCell>{record.sourceObject}</TableCell>
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={2} className="text-center py-8 text-gray-500">
                {objectRecords.length === 0 ? (
                  isLoadingObjectRecords ? (
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Loading records...
                    </div>
                  ) : (
                    <div>No objects found. Please check the data source.</div>
                  )
                ) : (
                  <div>No objects match the current filters.</div>
                )}
              </TableCell>
            </TableRow>
          )}
          {filteredRecords.length > 0 && (
            <TableRow>
              <TableCell colSpan={2}>
                <LoadMoreButton hasMore={hasMoreTokens} isLoading={isLoadingTokenRecords} onClick={onLoadMore ?? (() => {})} />

              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </>
    );
  }

  return (
    <>
      <TableHeader>
        <TableRow>
          <TableHead>Token</TableHead>
          <TableHead>Tag</TableHead>
          <TableHead>Context</TableHead>
          <TableHead>Source Object</TableHead>
          <TableHead>Groups</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {tokenRecords
          .filter((record) => {
            const tagMatches = tagFilters[record.tag];
            const groupMatches = record.groups.length === 0 || record.groups.some((group) => groupFilters[group]);
            return groupMatches && tagMatches;
          })
          .map((record, index) => (
            <TableRow key={index}>
              <TableCell>{record.token}</TableCell>
              <TableCell>
                <HighlightedTag tag={record.tag} tagColors={tagColors} />
              </TableCell>
              <TableCell className="max-w-sm">
                {record.context ? (
                  <TokenContext context={{
                    left: record.context.left,
                    right: record.context.right,
                    token: record.token,
                    tag: record.tag
                  }} tagColors={tagColors} />
                ) : (
                  <span className="text-red-400 text-xs">Missing context</span>
                )}
              </TableCell>
              <TableCell>{record.sourceObject}</TableCell>
              <TableCell>{record.groups.join(', ')}</TableCell>
            </TableRow>
          ))}
        {isLoadingTokenRecords && (
          <TableRow>
            <TableCell colSpan={5} className="text-center py-4 text-gray-500">
              <div className="flex items-center justify-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading more records...
              </div>
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </>
  );
}
