import { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import {
  DatabaseTableProps,
  ViewMode,
  ClassifiedTokenDatabaseRecord,
  ObjectDatabaseRecord
} from './types';
import { FilterSection } from './FilterSection';
import { HeaderContent } from './HeaderContent';
import { TableContent } from './TableContent';
import { nerService } from '@/lib/backend';
import { useSearchParams } from 'next/navigation';
import { NO_GROUP } from '@/lib/utils';

function joinAdjacentEntities(entities: Entity[]) {
  if (entities.length === 0) {
    return [];
  }

  const joinedEntities: Entity[] = [entities[0]];
  // We need to keep track of the previous *unjoined* entity.
  let prevEntity = entities[0];

  for (let i = 1; i < entities.length; i++) {
    console.log(entities[i]);
    const numSpacesAfterPrevEntity = (prevEntity.RContext || '').match(/^\s*/)?.[0].length || 0
    if (
      prevEntity.Object === entities[i].Object &&
      (entities[i].Start - prevEntity.End) === numSpacesAfterPrevEntity &&
      prevEntity.Label === entities[i].Label
    ) {
      const lastJoinedEntity = joinedEntities[joinedEntities.length - 1];
      const whitespace = prevEntity.RContext?.substring(0, numSpacesAfterPrevEntity) || '';
      joinedEntities[joinedEntities.length - 1] = {
        ...lastJoinedEntity,
        Text: lastJoinedEntity.Text + whitespace + entities[i].Text,
        End: entities[i].End,
        RContext: entities[i].RContext,
      }
    } else {
      joinedEntities.push(entities[i]);
    }
    prevEntity = entities[i];
  }

  return joinedEntities;
}


export function DatabaseTable({
  groups: groupsProp,
  tags
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
  const [tokenRecords, setTokenRecords] = useState<
    ClassifiedTokenDatabaseRecord[]
  >([]);
  const [objectRecords, setObjectRecords] = useState<ObjectDatabaseRecord[]>(
    []
  );
  const [viewMode, setViewMode] = useState<ViewMode>('classified-token');
  const [query, setQuery] = useState('');
  const [filteredObjects, setFilteredObjects] = useState<string[]>([]);

  // Pagination states
  const [tokenOffset, setTokenOffset] = useState(0);
  const [hasMoreTokens, setHasMoreTokens] = useState(true);
  const [objectOffset, setObjectOffset] = useState(0);
  const [hasMoreObjects, setHasMoreObjects] = useState(true);
  const TOKENS_LIMIT = 25; // Number of token records to fetch per request
  const OBJECTS_LIMIT = 10; // Number of object records to fetch per request

  // Filter states
  const [groupFilters, setGroupFilters] = useState<Record<string, boolean>>(
    () => Object.fromEntries(groups.map((group) => [group, true]))
  );

  const [tagFilters, setTagFilters] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(tags.map((tag) => [tag.type, true]))
  );

  // Load records functions
  const loadTokenRecords = (
    newOffset = 0,
    tagFilter?: string[],
    limit = TOKENS_LIMIT
  ) => {
    if (isLoadingTokenRecords || (!hasMoreTokens && newOffset > 0)) {
      console.log(
        'Skipping token records load - already loading or no more data'
      );
      return;
    }

    console.log(
      `Loading token records from offset=${newOffset}, tagFilter:`,
      tagFilter
    );
    setIsLoadingTokenRecords(true);

    nerService
      .getReportEntities(reportId, {
        offset: newOffset,
        limit: limit,
        ...(tagFilter && { tags: tagFilter })
      })
      .then((entities) => {
        console.log(
          `Loaded ${entities.length} token records from offset ${newOffset}`
        );

        const mappedRecords = joinAdjacentEntities(entities).map((entity) => ({
          token: entity.Text,
          tag: entity.Label,
          sourceObject: entity.Object,
          groups: [],
          context: {
            left: entity.LContext || '',
            right: entity.RContext || ''
          }
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

  const loadObjectRecords = (
    newOffset = 0,
    objectsFilter?: string[],
    limit = OBJECTS_LIMIT
  ) => {
    // Don't load if we're already loading or if we've reached the end
    if (isLoadingObjectRecords || (!hasMoreObjects && newOffset > 0)) {
      console.log(
        'Skipping object records load - already loading or no more data'
      );
      return;
    }

    console.log(`Loading object records from offset=${newOffset}`);
    setIsLoadingObjectRecords(true);

    // Use the API service to fetch objects with pagination
    nerService
      .getReportObjects(reportId, {
        offset: newOffset,
        limit: limit,
        ...(objectsFilter && { tags: objectsFilter })
      })
      .then((objects) => {
        console.log(
          `Loaded ${objects.length} object records from offset ${newOffset}`
        );

        // Map API objects to our record format
        const mappedRecords = objects.map((obj) => ({
          sourceObject: obj.object,
          taggedTokens: obj.tokens.map(
            (token, i) => [token, obj.tags[i]] as [string, string]
          ),
          groups: [] // This would need to be populated from somewhere if needed
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
      setFilteredObjects([]);
      resetPagination();
      loadTokenRecords(0);
      loadObjectRecords(0);
      return;
    }

    setIsSearching(true);
    try {
      const result = await nerService.searchReport(reportId, searchQuery);
      setFilteredObjects(result.Objects || []);

      // Reset pagination and clear existing records
      resetPagination();

      // Load records filtered by the object names
      if (result.Objects?.length) {
        loadObjectRecords(0, result.Objects);
        // For token records, we'll let them load unfiltered first
        // then filter them in the TableContent component
        loadTokenRecords(0);
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
    loadTokenRecords(0);
    loadObjectRecords(0);
    loadedInitialTokenRecords.current = true;
    loadedInitialObjectRecords.current = true;
  }, []);

  // Update filters when groups/tags change
  useEffect(() => {
    setGroupFilters((prev) => {
      const newFilters = { ...prev };
      let changed = false;

      groups.forEach((group) => {
        if (!(group in newFilters)) {
          newFilters[group] = true;
          changed = true;
        }
      });

      return changed ? newFilters : prev;
    });
  }, [groups]);

  useEffect(() => {
    setTagFilters((prev) => {
      const newFilters = { ...prev };
      let changed = false;

      tags.forEach((tag) => {
        if (!(tag.type in newFilters)) {
          newFilters[tag.type] = true;
          changed = true;
        }
      });

      return changed ? newFilters : prev;
    });
  }, [tags]);

  // Scroll handler for infinite loading
  const handleTableScroll = () => {
    if (tableScrollRef.current) {
      setShowTableShadow(tableScrollRef.current.scrollTop > 0);

      // Check if we're near the bottom
      const { scrollTop, scrollHeight, clientHeight } = tableScrollRef.current;
      const bottomThreshold = 200; // pixels from bottom to trigger load

      if (scrollHeight - (scrollTop + clientHeight) < bottomThreshold) {
        // Load more records based on view mode
        if (
          viewMode === 'object' &&
          !isLoadingObjectRecords &&
          hasMoreObjects
        ) {
          loadObjectRecords(
            objectOffset,
            filteredObjects.length ? filteredObjects : undefined
          );
        } else if (
          viewMode === 'classified-token' &&
          !isLoadingTokenRecords &&
          hasMoreTokens
        ) {
          loadTokenRecords(tokenOffset);
        }
      }
    }
  };

  // Load more handler for manual loading (can be used with a button)
  const handleLoadMore = () => {
    if (viewMode === 'object' && !isLoadingObjectRecords && hasMoreObjects) {
      loadObjectRecords(
        objectOffset,
        filteredObjects.length ? filteredObjects : undefined
      );
    } else if (
      viewMode === 'classified-token' &&
      !isLoadingTokenRecords &&
      hasMoreTokens
    ) {
      loadTokenRecords(tokenOffset);
    }
  };

  // Filter handlers
  const handleGroupFilterChange = (filterKey: string) => {
    setGroupFilters((prev) => ({
      ...prev,
      [filterKey]: !prev[filterKey]
    }));
  };

  const handleTagFilterChange = (filterKey: string) => {
    setTagFilters((prev) => ({
      ...prev,
      [filterKey]: !prev[filterKey]
    }));

    let activeTagsList = Object.entries(tagFilters)
      .filter(([_, isActive]) => isActive)
      .map(([tagName]) => tagName);

    activeTagsList.push(filterKey);

    setFilteredObjects([]);
    resetPagination();
    loadTokenRecords(0, activeTagsList, 100);
    loadObjectRecords(0, activeTagsList, 100);
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
    setFilteredObjects([]);
    resetPagination();
    loadTokenRecords(0);
    loadObjectRecords(0);

    setTagFilters(Object.fromEntries(tags.map((tag) => [tag.type, true])));
  };

  const handleDeselectAllTags = () => {
    setTagFilters(Object.fromEntries(tags.map((tag) => [tag.type, false])));
  };

  // Handle view mode changes
  const handleViewModeChange = (newMode: ViewMode) => {
    setViewMode(newMode);

    // Load data if we haven't loaded any for this view mode yet
    if (newMode === 'object' && objectRecords.length === 0) {
      loadObjectRecords(0);
    } else if (newMode === 'classified-token' && tokenRecords.length === 0) {
      loadTokenRecords(0);
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
      hasMoreObjects
    });
  }, [
    tokenRecords,
    objectRecords,
    tokenOffset,
    objectOffset,
    hasMoreTokens,
    hasMoreObjects
  ]);
  const [showFilterSection, setShowFilterSection] = useState<boolean>(true);

  return (
    <Card className="h-[70vh]">
      <CardContent className="p-0 h-full">
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
            onToggleFilterSection={() => { setShowFilterSection(!showFilterSection) }}
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
                boxShadow: showTableShadow
                  ? 'inset 0 4px 6px -4px rgba(0, 0, 0, 0.1)'
                  : 'none'
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
                />
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
