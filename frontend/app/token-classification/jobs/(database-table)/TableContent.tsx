import React, { useMemo } from 'react';
import { TableHead, TableRow, TableHeader, TableBody, TableCell } from '@/components/ui/table';
import { Loader2, ChevronRight } from 'lucide-react';
import { NO_GROUP } from '@/lib/utils';
import { TokenHighlighter } from '@/components/feedback/TokenHighlighter';
import * as _ from 'lodash';
import type { TableContentProps } from '@/types/analyticsTypes';
import { useLicense } from '@/hooks/useLicense';
import { environment } from '@/lib/environment';

const PASTELS = ['#E5A49C', '#F6C886', '#FBE7AA', '#99E3B5', '#A6E6E7', '#A5A1E1', '#D8A4E2'];
const DARKERS = ['#D34F3E', '#F09336', '#F7CF5F', '#5CC96E', '#65CFD0', '#597CE2', '#B64DC8'];

interface HighlightColor {
  text: string;
  tag: string;
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
  customTagNames,
  hasMoreTokens = false,
  hasMoreObjects = false,
  onLoadMore,
  showFilterContent,
  pathMap,
  addFeedback,
}: TableContentProps) {
  const { isEnterprise } = useLicense();

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
                    <TokenHighlighter
                      tokens={tokens}
                      availableTags={tags.map((tag) => tag.type)}
                      tagFilters={tagFilters}
                    />
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
    filterRecords(record.groups, [...new Set(record.taggedTokens.map((token) => token[1]))])
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
              tokens.map((token) =>
                !environment.allowCustomTagsInFeedback && customTagNames.includes(token.tag)
                  ? 'O'
                  : token.tag
              )
            );
          };
          return (
            <details
              key={index}
              className="group text-sm leading-relaxed bg-white rounded border border-gray-100 shadow-sm mb-4"
            >
              <summary className="p-3 cursor-pointer bg-gray-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 transition-transform group-open:rotate-90" />
                  {/* @ts-ignore */}
                  {fullPath && typeof window !== 'undefined' && window.electron ? (
                    <span className="font-semibold">{truncateFilePath(fullPath)}</span>
                  ) : (
                    <span
                      className="font-semibold"
                      style={{ color: 'inherit', userSelect: 'none' }}
                    >
                      {fileIdentifier.split('/').slice(-1)}
                    </span>
                  )}
                </div>

                {/* @ts-ignore */}
                {fullPath && typeof window !== 'undefined' && window.electron && (
                  <span
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      openFile();
                    }}
                    className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md bg-gray-50 text-sm font-medium text-gray-800 hover:bg-gray-100 hover:text-blue-600 border border-transparent hover:border-blue-200 transition-all cursor-pointer"
                  >
                    <span className="leading-none">OPEN</span>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="lucide lucide-external-link-icon lucide-external-link"
                    >
                      <path d="M15 3h6v6" />
                      <path d="M10 14 21 3" />
                      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                    </svg>
                  </span>
                )}
              </summary>

              <div className="p-4">
                <TokenHighlighter
                  tokens={tokens}
                  editable={isEnterprise}
                  availableTags={tags.map((tag) => tag.type)}
                  unassignableTags={!environment.allowCustomTagsInFeedback ? customTagNames : []}
                  onTagAssign={onTagAssign}
                  objectId={fileIdentifier}
                  tagFilters={tagFilters}
                />
                ...
                <br />
                <br />
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
