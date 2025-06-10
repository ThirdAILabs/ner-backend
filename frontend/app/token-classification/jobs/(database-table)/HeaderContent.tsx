import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface HeaderContentProps {
  viewMode: ViewMode;
  query: string;
  onQueryChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onViewModeChange: (value: ViewMode) => void;
  onSave: () => void;
  onSearch?: (query: string) => Promise<void>;
  searchLoading?: boolean;
}

export function HeaderContent({
  viewMode,
  query,
  onQueryChange,
  onViewModeChange,
  onSave,
  onSearch,
  searchLoading = false,
}: HeaderContentProps) {
  // Cast the ViewMode to string for compatibility with the Tabs component
  const handleViewModeChange = (value: string) => {
    onViewModeChange(value as ViewMode);
  };

  const handleSearch = () => {
    if (onSearch) {
      onSearch(query);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && onSearch) {
      onSearch(query);
    }
  };

  return (
    <div className="p-6 pb-2 pt-4">
      <div className="flex items-center space-x-4">
        <div className="font-medium">View By</div>
        <Tabs value={viewMode} onValueChange={handleViewModeChange}>
          <TabsList>
            <TabsTrigger value="object">File</TabsTrigger>
            <TabsTrigger value="classified-token">Token</TabsTrigger>
          </TabsList>
        </Tabs>
        {/* <div className="font-medium pl-2">Query</div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <div className="flex-1">
              <Input
                type="text"
                placeholder="Enter query..."
                value={query}
                onChange={onQueryChange}
                onKeyDown={handleKeyDown}
              />
            </div>
            <Button
              onClick={handleSearch}
              disabled={searchLoading || !onSearch}
              className="h-10 w-10 p-2"
              variant="outline"
            >
              {searchLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Search className="h-5 w-5" />
              )}
            </Button>
            <SaveButton
              onClick={onSave}
              style={{
                width: '40px',
                height: '40px',
                minWidth: '40px',
                padding: '8px',
              }}
            />
          </div>
        </div> */}
      </div>
    </div>
  );
}
