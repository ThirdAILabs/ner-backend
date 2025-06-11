import { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { FilterSection } from './FilterSection';
import { HeaderContent } from './HeaderContent';
import { TableContent } from './TableContent';
import { nerService } from '@/lib/backend';
import { useSearchParams } from 'next/navigation';
import { NO_GROUP } from '@/lib/utils';

export function DatabaseTable({
  groups: groupsProp,
  tags,
  uploadId,
  initialSelectedTag,
}: DatabaseTableProps) {
  const searchParams = useSearchParams();
  const reportId: string = searchParams.get('jobId') as string;
  const groups = groupsProp.length > 0 ? [...groupsProp, NO_GROUP] : [];

  // Loading states and refs
  const [isLoadingTokenRecords, setIsLoadingTokenRecords] = useState(false);
  const [isLoadingObjectRecords, setIsLoadingObjectRecords] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const loadedInitialTokenRecords = useRef(false);
  const loadedInitialObjectRecords = useRef(false);
  const tableScrollRef = useRef<HTMLDivElement>(null);
  const [showTableShadow, setShowTableShadow] = useState(false);

  // Data states
  const [tokenRecords, setTokenRecords] = useState<ClassifiedTokenDatabaseRecord[]>([]);
  const [objectRecords, setObjectRecords] = useState<ObjectDatabaseRecord[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('object');
  const [query, setQuery] = useState('');
  const [pathMap, setPathMap] = useState<Record<string, string>>({});

  // Pagination states
  const [tokenOffset, setTokenOffset] = useState(0);
  const [hasMoreTokens, setHasMoreTokens] = useState(true);
  const [objectOffset, setObjectOffset] = useState(0);
  const [hasMoreObjects, setHasMoreObjects] = useState(true);
  const TOKENS_LIMIT = 25; // Number of token records to fetch per request
  const OBJECTS_LIMIT = 25; // Number of object records to fetch per request

  // Filter states
  const [groupFilters, setGroupFilters] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(groups.map((group) => [group, true]))
  );

  const [tagFilters, setTagFilters] = useState<Record<string, boolean>>(() => {
    const allTagKeys = tags.map((tag) => tag.type);
    return Object.fromEntries(
      allTagKeys.map((tag) => [tag, initialSelectedTag ? tag === initialSelectedTag : true])
    );
  });

  const toActiveTagList = (filters: Record<string, boolean>): string[] => {
    return Object.entries(filters)
      .filter(([_, isActive]) => isActive)
      .map(([tagType]) => tagType);
  };

  // Load records functions
  const loadTokenRecords = (newOffset = 0, tagFilter: string[], limit = TOKENS_LIMIT) => {
    if (isLoadingTokenRecords || (!hasMoreTokens && newOffset > 0)) {
      console.log('Skipping token records load - already loading or no more data');
      return;
    }

    setIsLoadingTokenRecords(true);

    nerService
      .getReportEntities(reportId, {
        offset: newOffset,
        limit: limit,
        tags: tagFilter,
      })
      .then((entities) => {
        console.log(`Loaded ${entities.length} token records from offset ${newOffset}`);

        const mappedRecords = entities.map((entity) => ({
          token: entity.Text,
          tag: entity.Label,
          sourceObject: entity.Object,
          groups: [],
          context: {
            left: entity.LContext || '',
            right: entity.RContext || '',
          },
          start: entity.Start,
          end: entity.End,
        }));

        if (newOffset === 0) {
          setTokenRecords(mappedRecords);
        } else {
          setTokenRecords((prev) => [...prev, ...mappedRecords]);
        }

        setHasMoreTokens(entities.length === TOKENS_LIMIT);
        setTokenOffset(newOffset + entities.length);
        setIsLoadingTokenRecords(false);
      })
      .catch((error) => {
        console.error('Error loading token records:', error);
        setIsLoadingTokenRecords(false);
      });
  };

  const loadObjectRecords = (newOffset = 0, objectsFilter?: string[], limit = OBJECTS_LIMIT) => {
    // Don't load if we're already loading or if we've reached the end
    if (isLoadingObjectRecords || (!hasMoreObjects && newOffset > 0)) {
      console.log('Skipping object records load - already loading or no more data');
      return;
    }

    setIsLoadingObjectRecords(true);

    // Use the API service to fetch objects with pagination
    nerService
      .getReportObjects(reportId, {
        offset: newOffset,
        limit: limit,
        tags: objectsFilter,
      })
      .then((objects) => {
        console.log(`Loaded ${objects.length} object records from offset ${newOffset}`);

        // Map API objects to our record format
        const mappedRecords = objects.map((obj) => ({
          sourceObject: obj.object,
          taggedTokens:
            obj.tokens?.map((token, i) => [token, obj.tags[i]] as [string, string]) || [],
          groups: [], // This would need to be populated from somewhere if needed
        }));

        // If resetting (offset=0), replace records; otherwise append
        if (newOffset === 0) {
          setObjectRecords(mappedRecords);
        } else {
          setObjectRecords((prev) => [...prev, ...mappedRecords]);
        }

        // Update pagination state
        setHasMoreObjects(objects.length === OBJECTS_LIMIT);
        setObjectOffset(newOffset + objects.length);
        setIsLoadingObjectRecords(false);
      })
      .catch((error) => {
        console.error('Error loading object records:', error);
        setIsLoadingObjectRecords(false);
      });
  };

  // Search function
  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      // If empty query, reset filters and reload data
      resetPagination();
      loadTokenRecords(0, toActiveTagList(tagFilters));
      loadObjectRecords(0, toActiveTagList(tagFilters));
      return;
    }

    setIsSearching(true);
    try {
      const result = await nerService.searchReport(reportId, searchQuery);
      // Reset pagination and clear existing records
      resetPagination();

      // Load records filtered by the object names
      if (result.Objects?.length) {
        loadObjectRecords(0, toActiveTagList(tagFilters));
        // For token records, we'll let them load unfiltered first
        // then filter them in the TableContent component
        loadTokenRecords(0, toActiveTagList(tagFilters));
      }
    } catch (error) {
      console.error('Error searching:', error);
    } finally {
      setIsSearching(false);
    }
  };

  // Helper function to reset pagination state
  const resetPagination = () => {
    setTokenOffset(0);
    setObjectOffset(0);
    setHasMoreTokens(true);
    setHasMoreObjects(true);
    setTokenRecords([]);
    setObjectRecords([]);
  };

  // Initial load
  useEffect(() => {
    // Force immediate loading of token records
    console.log('Component mounted, loading initial data');
    resetPagination();
    loadTokenRecords(0, toActiveTagList(tagFilters));
    loadObjectRecords(0, toActiveTagList(tagFilters));
    loadedInitialTokenRecords.current = true;
    loadedInitialObjectRecords.current = true;
  }, []);

  // Load path map
  useEffect(() => {
    if (uploadId) {
      nerService
        .getFileNameToPath(uploadId)
        .then((pathMap) => {
          setPathMap(pathMap);
        })
        .catch((error) => {
          console.error('Could not load path map:', error);
        });
    }
  }, [uploadId]);

  // Scroll handler for infinite loading
  const handleTableScroll = () => {
    if (tableScrollRef.current) {
      setShowTableShadow(tableScrollRef.current.scrollTop > 0);

      // Check if we're near the bottom
      const { scrollTop, scrollHeight, clientHeight } = tableScrollRef.current;
      const bottomThreshold = 200; // pixels from bottom to trigger load

      if (scrollHeight - (scrollTop + clientHeight) < bottomThreshold) {
        // Load more records based on view mode
        if (viewMode === 'object' && !isLoadingObjectRecords && hasMoreObjects) {
          loadObjectRecords(objectOffset, toActiveTagList(tagFilters));
        } else if (viewMode === 'classified-token' && !isLoadingTokenRecords && hasMoreTokens) {
          loadTokenRecords(tokenOffset, toActiveTagList(tagFilters));
        }
      }
    }
  };

  // Load more handler for manual loading (can be used with a button)
  const handleLoadMore = () => {
    if (viewMode === 'object' && !isLoadingObjectRecords && hasMoreObjects) {
      loadObjectRecords(objectOffset, toActiveTagList(tagFilters));
    } else if (viewMode === 'classified-token' && !isLoadingTokenRecords && hasMoreTokens) {
      loadTokenRecords(tokenOffset, toActiveTagList(tagFilters));
    }
  };

  // Filter handlers
  const handleGroupFilterChange = (filterKey: string) => {
    setGroupFilters((prev) => ({
      ...prev,
      [filterKey]: !prev[filterKey],
    }));
  };

  const handleTagFilterChange = (filterKey: string) => {
    setTagFilters((prev) => {
      const newFilters = {
        ...prev,
        [filterKey]: !prev[filterKey],
      };

      const activeTagList = toActiveTagList(newFilters);

      resetPagination();
      loadTokenRecords(0, activeTagList);
      loadObjectRecords(0, activeTagList);

      return newFilters;
    });
  };

  const handleSelectAllGroups = () => {
    console.log('Selecting all groups');
    setGroupFilters(Object.fromEntries(groups.map((group) => [group, true])));
  };

  const handleDeselectAllGroups = () => {
    console.log('Deselecting all groups');
    setGroupFilters(Object.fromEntries(groups.map((group) => [group, false])));
  };

  const handleSelectAllTags = () => {
    const newFilters = Object.fromEntries(tags.map((tag) => [tag.type, true]));
    setTagFilters(newFilters);
    resetPagination();
    loadTokenRecords(0, toActiveTagList(newFilters));
    loadObjectRecords(0, toActiveTagList(newFilters));
  };

  const handleDeselectAllTags = () => {
    // We need to explicitly set the objectRecords & tokenRecords to empty arrays (resetPagination handles that).
    // If we reuse the useEffect logic, it will set all the filters to false, which are then
    // ignored and not sent to the backend, so the backend assumes that there are no filters
    // and returns all the records.
    const newFilters = Object.fromEntries(tags.map((tag) => [tag.type, false]));
    setTagFilters(newFilters);
    resetPagination();
  };

  // Handle view mode changes
  const handleViewModeChange = (newMode: ViewMode) => {
    setViewMode(newMode);

    // Load data if we haven't loaded any for this view mode yet
    if (newMode === 'object' && objectRecords.length === 0) {
      loadObjectRecords(0, toActiveTagList(tagFilters));
    } else if (newMode === 'classified-token' && tokenRecords.length === 0) {
      loadTokenRecords(0, toActiveTagList(tagFilters));
    }
  };

  // Other handlers
  const handleQueryChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(event.target.value);
  };

  const handleSave = () => {
    console.log('Saving...');
  };

  // Add a debug effect to monitor record state
  useEffect(() => {
    console.log('Records updated:', {
      tokenRecords: tokenRecords.length,
      tokenOffset,
      hasMoreTokens,
      objectRecords: objectRecords.length,
      objectOffset,
      hasMoreObjects,
    });
  }, [tokenRecords, objectRecords, tokenOffset, objectOffset, hasMoreTokens, hasMoreObjects]);
  const [showFilterSection, setShowFilterSection] = useState<boolean>(true);

  return (
    <Card className="h-[79vh]">
      <CardContent className="pb-2 h-full">
        <div className="flex h-full">
          <FilterSection
            groups={groups}
            tags={tags}
            groupFilters={groupFilters}
            tagFilters={tagFilters}
            showFilterSection={showFilterSection}
            onGroupFilterChange={handleGroupFilterChange}
            onTagFilterChange={handleTagFilterChange}
            onSelectAllGroups={handleSelectAllGroups}
            onDeselectAllGroups={handleDeselectAllGroups}
            onSelectAllTags={handleSelectAllTags}
            onDeselectAllTags={handleDeselectAllTags}
            onToggleFilterSection={() => {
              setShowFilterSection((prev) => !prev);
            }}
          />

          <div className="flex-1 flex flex-col h-full">
            <HeaderContent
              viewMode={viewMode}
              query={query}
              onQueryChange={handleQueryChange}
              onViewModeChange={handleViewModeChange}
              onSave={handleSave}
              onSearch={handleSearch}
              searchLoading={isSearching}
            />

            <div
              ref={tableScrollRef}
              className="flex-1 overflow-auto"
              onScroll={handleTableScroll}
              style={{
                boxShadow: showTableShadow ? 'inset 0 4px 6px -4px rgba(0, 0, 0, 0.1)' : 'none',
              }}
            >
              <div className="px-6">
                <TableContent
                  viewMode={viewMode}
                  objectRecords={objectRecords}
                  tokenRecords={tokenRecords}
                  groupFilters={groupFilters}
                  tagFilters={tagFilters}
                  isLoadingObjectRecords={isLoadingObjectRecords}
                  isLoadingTokenRecords={isLoadingTokenRecords}
                  tags={tags}
                  hasMoreTokens={hasMoreTokens}
                  hasMoreObjects={hasMoreObjects}
                  onLoadMore={handleLoadMore}
                  showFilterContent={showFilterSection}
                  pathMap={pathMap}
                />
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
