import { useRef, useState } from 'react';
import { ChevronDown, ChevronUp, ListFilter } from 'lucide-react';
import type { Tag } from '@/types/analyticsTypes';
import { SidePanel } from '@/components/ui/collapsible/sidepanel';

interface FilterSectionProps {
  groups: string[];
  tags: Tag[];
  groupFilters: Record<string, boolean>;
  tagFilters: Record<string, boolean>;
  showFilterSection: boolean;
  onGroupFilterChange: (filterKey: string) => void;
  onTagFilterChange: (filterKey: string) => void;
  onSelectAllGroups: () => void;
  onDeselectAllGroups: () => void;
  onSelectAllTags: () => void;
  onDeselectAllTags: () => void;
  onToggleFilterSection: () => void;
}

export function FilterSection({
  groups,
  tags,
  groupFilters,
  tagFilters,
  showFilterSection,
  onGroupFilterChange,
  onTagFilterChange,
  onSelectAllGroups,
  onDeselectAllGroups,
  onSelectAllTags,
  onDeselectAllTags,
  onToggleFilterSection
}: FilterSectionProps) {
  const [isGroupsExpanded, setIsGroupsExpanded] = useState(true);
  const [isTagsExpanded, setIsTagsExpanded] = useState(true);

  const toggleGroups = () => setIsGroupsExpanded(!isGroupsExpanded);
  const toggleTags = () => setIsTagsExpanded(!isTagsExpanded);

  return (
    <SidePanel
      width={280}
      showPanel={showFilterSection}
      onTogglePanel={onToggleFilterSection}
      borderPosition="right"
      expandedOffset="12px"
      collapsedOffset="-8px"
      className="bg-white"
    >
      {/* Fixed Filter Header */}
      <div className="sticky top-0 p-6 pb-2 pt-4 z-10">
        <div className="flex items-center gap-2">
          <ListFilter className="h-5 w-5" />
          <span className="flex font-medium h-[40px] items-center">Filter</span>
        </div>
      </div>

      {/* Filter Content */}
      <div className="p-6 pt-4 space-y-6">
        {/* Groups Section */}
        <div>
          <div
            className="flex items-center justify-between text-sm text-gray-600 mb-2 cursor-pointer hover:text-gray-800"
            onClick={toggleGroups}
          >
            <span>Groups</span>
            {isGroupsExpanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </div>
          {isGroupsExpanded && groups.length > 0 && (
            <>
              <div className="flex gap-2 mb-2">
                <button
                  onClick={onSelectAllGroups}
                  className="text-xs text-blue-600 hover:text-blue-800"
                >
                  Select All
                </button>
                <span className="text-gray-300">|</span>
                <button
                  onClick={onDeselectAllGroups}
                  className="text-xs text-blue-600 hover:text-blue-800"
                >
                  Deselect All
                </button>
              </div>
              <div className="space-y-2">
                {groups.map((filter) => (
                  <label key={filter} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={groupFilters[filter]}
                      onChange={() => onGroupFilterChange(filter)}
                      className="mr-2"
                    />
                    {filter}
                  </label>
                ))}
              </div>
            </>
          )}
          {groups.length === 0 && (
            <div className="text-gray-500 text-sm">No groups configured.</div>
          )}
        </div>

        {/* Tags Section */}
        <div>
          <div
            className="flex items-center justify-between text-sm text-gray-600 mb-2 cursor-pointer hover:text-gray-800"
            onClick={toggleTags}
          >
            <span>Tags</span>
            {isTagsExpanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </div>
          {isTagsExpanded && (
            <>
              <div className="flex gap-2 mb-2">
                <button
                  onClick={onSelectAllTags}
                  className="text-xs text-blue-600 hover:text-blue-800"
                >
                  Select All
                </button>
                <span className="text-gray-300">|</span>
                <button
                  onClick={onDeselectAllTags}
                  className="text-xs text-blue-600 hover:text-blue-800"
                >
                  Deselect All
                </button>
              </div>
              <div className="space-y-2">
                {[...tags]
                  .filter((filter) => filter.count > 0)
                  .sort((a, b) => b.count - a.count)
                  .map((filter) => (
                    <label key={filter.type} className="flex items-center">
                      <input
                        type="checkbox"
                        checked={tagFilters[filter.type]}
                        onChange={() => onTagFilterChange(filter.type)}
                        className="mr-2"
                      />
                      {filter.type} ({filter.count})
                    </label>
                  ))}
              </div>
            </>
          )}
        </div>
      </div>
    </SidePanel>
  );
}
