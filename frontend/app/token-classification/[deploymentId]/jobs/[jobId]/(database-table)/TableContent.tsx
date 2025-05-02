import {
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
} from '@/components/ui/table';
import { Loader2 } from 'lucide-react';
import { ClassifiedTokenDatabaseRecord, ObjectDatabaseRecord, ViewMode } from './types';
import { useMemo } from 'react';

// Define color palettes for tag colors
const PASTELS = ['#E5A49C', '#F6C886', '#FBE7AA', '#99E3B5', '#A6E6E7', '#A5A1E1', '#D8A4E2'];
const DARKERS = ['#D34F3E', '#F09336', '#F7CF5F', '#5CC96E', '#65CFD0', '#597CE2', '#B64DC8'];

// Default color for tags not in the computed colors map
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

function HighlightedToken({ token, tag, tagColors }: HighlightedTokenProps) {
  // If tag is 'O', return the token with spacing but no highlighting
  if (tag === 'O') {
    return (
      <span style={{ marginRight: '4px' }}>
        {token}
      </span>
    );
  }

  const tagColor = tagColors[tag] || DEFAULT_COLOR;

  return (
    <span
      style={{
        backgroundColor: tagColor.text,
        padding: '2px',
        borderRadius: '2px',
        userSelect: 'none',
        display: 'inline-flex',
        alignItems: 'center',
        marginRight: '4px',
        wordBreak: 'break-word'
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
  );
}

// Add a new component for highlighting tags
function HighlightedTag({ tag, tagColors }: { tag: string; tagColors: Record<string, HighlightColor> }) {
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

// Add a new component for context display
interface TokenContextProps {
  context?: { left: string; right: string; token: string; tag: string; };
  tagColors: Record<string, HighlightColor>;
}

function TokenContext({ context, tagColors }: TokenContextProps) {
  if (!context) return <span className="text-red-400 text-xs">No context available</span>;
  console.log("context", context);
  // Add debug information
  const leftContent = context.left || '[empty left context]';
  const rightContent = context.right || '[empty right context]';
  const token = context.token || '[empty token]';
  const tag = context.tag || '[empty tag]';
  return (
    <div className="font-mono text-xs border border-gray-200 p-2 rounded bg-gray-50">
      <span className="text-gray-600">{leftContent}</span>
      {/* <span className="font-bold px-1 mx-1 text-black bg-yellow-200 rounded">
        <span className="text-black">«TOKEN»</span>
      </span> */}
      <HighlightedToken
        token={token}
        tag={tag}
        tagColors={tagColors}
      />
      <span className="text-gray-600">{rightContent}</span>
    </div>
  );
}

interface TableContentProps {
  viewMode: ViewMode;
  objectRecords: ObjectDatabaseRecord[];
  tokenRecords: ClassifiedTokenDatabaseRecord[];
  groupFilters: Record<string, boolean>;
  tagFilters: Record<string, boolean>;
  isLoadingObjectRecords: boolean;
  isLoadingTokenRecords: boolean;
  tags: string[]; // Add tags prop
}

export function TableContent({
  viewMode,
  objectRecords,
  tokenRecords,
  groupFilters,
  tagFilters,
  isLoadingObjectRecords,
  isLoadingTokenRecords,
  tags,
}: TableContentProps) {
  // Debug log for records
  console.log("TableContent received:", {
    viewMode,
    tokenRecordsCount: tokenRecords.length,
    objectRecordsCount: objectRecords.length,
    tagFilters,
    groupFilters
  });

  // Compute tag colors based on the provided tags
  const tagColors = useMemo(() => {
    const colors: Record<string, HighlightColor> = {};

    // Filter out 'O' tag and assign colors to each tag
    tags
      .filter(tag => tag !== 'O')
      .forEach((tag, index) => {
        colors[tag] = {
          text: PASTELS[index % PASTELS.length],
          tag: DARKERS[index % DARKERS.length],
        };
      });

    return colors;
  }, [tags]);

  // For classified token view
  if (viewMode === 'classified-token') {
    // Filter records based on criteria
    const filteredRecords = tokenRecords.filter((record) => {
      // Check if the tag is in the filter or if filter is undefined for this tag type
      // This handles the case where API returns tag types not in our mockTags list
      const tagMatches = tagFilters[record.tag] !== false; // Consider it a match unless explicitly set to false

      // If there are no groups in the record, consider it a match
      // Otherwise, check if at least one group matches the filter
      const groupMatches = record.groups.length === 0 ||
        record.groups.some((group) => groupFilters[group] !== false);

      return groupMatches && tagMatches;
    });

    console.log("Filtered token records:", filteredRecords.length);
    console.log("Sample record:", filteredRecords.length > 0 ? filteredRecords[0] : "No records match filters");

    return (
      <>
        <TableHeader>
          <TableRow>
            <TableHead>Prediction</TableHead>
            <TableHead>Source Object</TableHead>
            {/* <TableHead>Groups</TableHead> */}
          </TableRow>
        </TableHeader>
        <TableBody>
          {filteredRecords.length > 0 ? (
            filteredRecords.map((record, index) => (
              <TableRow key={index}>
                <TableCell

                >
                  {record.context ? (
                    <TokenContext context={{
                      left: record.context?.left,
                      right: record.context?.right,
                      token: record.token,
                      tag: record.tag
                    }} tagColors={tagColors} />
                  ) : (
                    <span className="text-red-400 text-xs">Missing context</span>
                  )}
                </TableCell>
                <TableCell >{record.sourceObject}</TableCell>
                {/* <TableCell className="w-20 truncate">{record.groups.join(', ')}</TableCell> */}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={5} className="text-center py-8 text-gray-500">
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
          {isLoadingTokenRecords && filteredRecords.length > 0 && (
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

  // For object view
  if (viewMode === 'object') {
    return (
      <>
        <TableHeader>
          <TableRow>
            <TableHead>Tagged Tokens</TableHead>
            <TableHead>Source Object</TableHead>
            {/* <TableHead>Groups</TableHead> */}
          </TableRow>
        </TableHeader>
        <TableBody>
          {objectRecords
            .filter((record) => {
              // If there are no groups in the record, consider it a match
              // Otherwise, check if at least one group matches the filter
              const groupMatches = record.groups.length === 0 ||
                record.groups.some((group) => groupFilters[group]);

              return groupMatches;
            })
            .map((record, index) => (
              <TableRow key={index}>
                <TableCell
                  style={{
                    maxWidth: '50%',
                    whiteSpace: 'normal',
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word',
                  }}
                >
                  {record.taggedTokens.map((token, tokenIndex) => (
                    <HighlightedToken
                      key={`${index}-${tokenIndex}`}
                      token={token[0]}
                      tag={tagFilters[token[1]] ? token[1] : "O"}
                      tagColors={tagColors}
                    />
                  ))}
                </TableCell>
                <TableCell>{record.sourceObject}</TableCell>
                {/* <TableCell>{record.groups.join(', ')}</TableCell> */}
              </TableRow>
            ))}
          {isLoadingObjectRecords && (
            <TableRow>
              <TableCell colSpan={3} className="text-center py-4 text-gray-500">
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
            // Check if the tag is in the filter
            const tagMatches = tagFilters[record.tag];

            // If there are no groups in the record, consider it a match
            // Otherwise, check if at least one group matches the filter
            const groupMatches = record.groups.length === 0 ||
              record.groups.some((group) => groupFilters[group]);

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
                    left: record.context?.left,
                    right: record.context?.right,
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