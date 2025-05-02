import { useEffect, useRef, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Table } from '@/components/ui/table';
import { DatabaseTableProps, ViewMode, ClassifiedTokenDatabaseRecord, ObjectDatabaseRecord } from './types';
import { FilterSection } from './FilterSection';
import { HeaderContent } from './HeaderContent';
import { TableContent } from './TableContent';
import { nerService } from '@/lib/backend';
import { useParams } from 'next/navigation';

export function DatabaseTable({
  loadMoreObjectRecords,
  loadMoreClassifiedTokenRecords,
  groups,
  tags,
}: DatabaseTableProps) {
  const params = useParams();
  const reportId: string = params.jobId as string;
  
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
  const [viewMode, setViewMode] = useState<ViewMode>('classified-token');
  const [query, setQuery] = useState('');
  const [filteredObjects, setFilteredObjects] = useState<string[]>([]);

  // Filter states
  const [groupFilters, setGroupFilters] = useState<Record<string, boolean>>(
    Object.fromEntries(groups.map((group) => [group, true]))
  );
  const [tagFilters, setTagFilters] = useState<Record<string, boolean>>(
    Object.fromEntries(tags.map((tag) => [tag, true]))
  );

  // Load records functions
  const loadTokenRecords = (objectFilter?: string) => {
    console.log("Loading token records, objectFilter:", objectFilter);
    setIsLoadingTokenRecords(true);
    loadMoreClassifiedTokenRecords()
      .then((records) => {
        console.log("Loaded token records:", records.length);
        const filtered = objectFilter 
          ? records.filter(record => objectFilter === record.sourceObject)
          : records;
        setTokenRecords((prev) => {
          const newRecords = [...prev, ...filtered];
          console.log("Total token records after update:", newRecords.length);
          return newRecords;
        });
        setIsLoadingTokenRecords(false);
      })
      .catch(error => {
        console.error("Error loading token records:", error);
        setIsLoadingTokenRecords(false);
      });
  };

  const loadObjectRecords = (objectsFilter?: string[]) => {
    setIsLoadingObjectRecords(true);
    // This uses the updated loadMoreObjectRecords which fetches data from /reports/{report_id}/objects
    // Now returning complete sentences with tagged tokens for better readability
    loadMoreObjectRecords().then((records) => {
      const filtered = objectsFilter?.length 
        ? records.filter(record => objectsFilter.includes(record.sourceObject))
        : records;
      setObjectRecords((prev) => [...prev, ...filtered]);
      setIsLoadingObjectRecords(false);
    });
  };

  // Search function
  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      // If empty query, reset filters and reload data
      setFilteredObjects([]);
      setTokenRecords([]);
      setObjectRecords([]);
      loadObjectRecords();
      loadTokenRecords();
      return;
    }

    setIsSearching(true);
    try {
      const result = await nerService.searchReport(reportId, searchQuery);
      setFilteredObjects(result.Objects || []);
      
      // Clear existing records and load new ones with filter
      setTokenRecords([]);
      setObjectRecords([]);
      
      // Load records filtered by the object names
      if (result.Objects?.length) {
        loadObjectRecords(result.Objects);
        loadTokenRecords();
      }
    } catch (error) {
      console.error("Error searching:", error);
    } finally {
      setIsSearching(false);
    }
  };

  // Initial load
  useEffect(() => {
    // Force immediate loading of token records
    console.log("Component mounted, loading initial data");
    setTokenRecords([]);
    setObjectRecords([]);
    loadTokenRecords();
    loadObjectRecords();
    loadedInitialTokenRecords.current = true;
    loadedInitialObjectRecords.current = true;
  }, []);

  // Update filters when groups/tags change
  useEffect(() => {
    setGroupFilters(Object.fromEntries(groups.map((group) => [group, true])));
  }, [groups]);

  useEffect(() => {
    setTagFilters(Object.fromEntries(tags.map((tag) => [tag, true])));
  }, [tags]);

  // Scroll handler for infinite loading
  const handleTableScroll = () => {
    if (tableScrollRef.current) {
      setShowTableShadow(tableScrollRef.current.scrollTop > 0);

      // Check if we're near the bottom
      const { scrollTop, scrollHeight, clientHeight } = tableScrollRef.current;
      const bottomThreshold = 100; // pixels from bottom to trigger load
      
      if (scrollHeight - (scrollTop + clientHeight) < bottomThreshold) {
        // Load more records based on view mode
        if (viewMode === 'object' && !isLoadingObjectRecords) {
          loadObjectRecords(filteredObjects.length ? filteredObjects : undefined);
        } else if (viewMode === 'classified-token' && !isLoadingTokenRecords) {
          loadTokenRecords();
        }
      }
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
    setTagFilters((prev) => ({
      ...prev,
      [filterKey]: !prev[filterKey],
    }));
  };

  const handleSelectAllGroups = () => {
    setGroupFilters(Object.fromEntries(groups.map((group) => [group, true])));
  };

  const handleDeselectAllGroups = () => {
    setGroupFilters(Object.fromEntries(groups.map((group) => [group, false])));
  };

  const handleSelectAllTags = () => {
    setTagFilters(Object.fromEntries(tags.map((tag) => [tag, true])));
  };

  const handleDeselectAllTags = () => {
    setTagFilters(Object.fromEntries(tags.map((tag) => [tag, false])));
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
    console.log("Records updated:", {
      tokenRecords: tokenRecords.length,
      objectRecords: objectRecords.length
    });
  }, [tokenRecords, objectRecords]);

  return (
    <Card className="h-[70vh]">
      <CardContent className="p-0 h-full">
        <div className="flex h-full">
          <FilterSection
            groups={groups}
            tags={tags}
            groupFilters={groupFilters}
            tagFilters={tagFilters}
            onGroupFilterChange={handleGroupFilterChange}
            onTagFilterChange={handleTagFilterChange}
            onSelectAllGroups={handleSelectAllGroups}
            onDeselectAllGroups={handleDeselectAllGroups}
            onSelectAllTags={handleSelectAllTags}
            onDeselectAllTags={handleDeselectAllTags}
          />

          <div className="flex-1 flex flex-col h-full">
            <HeaderContent
              viewMode={viewMode}
              query={query}
              onQueryChange={handleQueryChange}
              onViewModeChange={setViewMode}
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
                  : 'none',
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
                />
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 