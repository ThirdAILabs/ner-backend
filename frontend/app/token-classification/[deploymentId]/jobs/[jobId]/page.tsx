'use client';

import React, { useEffect } from 'react';
import { useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { CheckCircle, ArrowLeft, RefreshCw, Pause, Square, Plus } from 'lucide-react';
import { AnalyticsDashboard } from '@/components/AnalyticsDashboard';
import { DatabaseTable } from './(database-table)/DatabaseTable';
import { nerService } from '@/lib/backend';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";

// Calculate progress based on InferenceTaskStatuses
const calculateProgress = (report: Report | null): number => {
  if (!report || !report.InferenceTaskStatuses) return 0;

  const statuses = report.InferenceTaskStatuses;

  // Sum up all task sizes
  let totalSize = 0;
  let completedSize = 0;

  // Add completed tasks
  if (statuses.COMPLETED) {
    totalSize += statuses.COMPLETED.TotalSize;
    completedSize += statuses.COMPLETED.TotalSize;
  }

  // Add running tasks
  if (statuses.RUNNING) {
    totalSize += statuses.RUNNING.TotalSize;
  }

  // Add queued tasks
  if (statuses.QUEUED) {
    totalSize += statuses.QUEUED.TotalSize;
  }

  // Add failed tasks
  if (statuses.FAILED) {
    totalSize += statuses.FAILED.TotalSize;
  }

  // Calculate percentage
  if (totalSize === 0) return 0;
  return Math.round((completedSize / totalSize) * 100);
};

// Get the total number of processed tokens
const getProcessedTokens = (report: Report | null): number => {
  if (!report || !report.InferenceTaskStatuses || !report.InferenceTaskStatuses.COMPLETED) {
    return 0;
  }

  return report.InferenceTaskStatuses.COMPLETED.TotalSize;
};

// Mock data for database table
const mockGroups = ['Reject', 'Sensitive', 'Safe'];

// Source option card component
interface SourceOptionProps {
  title: string;
  description: string;
  isSelected?: boolean;
  disabled?: boolean;
  onClick: () => void;
}

const SourceOption: React.FC<SourceOptionProps> = ({
  title,
  description,
  isSelected = false,
  disabled = false,
  onClick
}) => (
  <div
    className={`relative p-6 border rounded-md transition-all
      ${isSelected ? 'border-blue-500 border-2' : 'border-gray-200'}
      ${disabled
        ? 'opacity-50 cursor-not-allowed bg-gray-50'
        : 'cursor-pointer hover:border-blue-300'
      }
    `}
    onClick={() => !disabled && onClick()}
  >
    <h3 className="text-base font-medium">{title}</h3>
    <p className="text-sm text-gray-500 mt-1">{description}</p>
  </div>
);

// Tag chip component
interface TagProps {
  tag: string;
  selected?: boolean;
  onClick?: () => void;
  custom?: boolean;
  addNew?: boolean;
  displayOnly?: boolean;
}

const Tag: React.FC<TagProps> = ({ tag, selected = true, onClick, custom = false, addNew = false, displayOnly = false }) => {
  return (
    <div
      className={`px-3 py-1 text-sm font-medium rounded-sm ${displayOnly ? "bg-blue-100 text-blue-800" : selected ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"} ${!displayOnly && onClick ? "cursor-pointer" : ""}`}
      style={{ userSelect: 'none' }}
      onClick={displayOnly ? undefined : onClick}
    >
      {tag}
    </div>
  );
}

// Group card component
interface GroupProps {
  name: string;
  definition: string;
}

const GroupCard: React.FC<GroupProps> = ({ name, definition }) => (
  <div className="border border-gray-200 rounded-md overflow-hidden">
    <div className="p-4 border-b border-gray-200">
      <h3 className="text-base font-medium">{name}</h3>
    </div>
    <div className="p-4">
      <p className="text-sm font-mono">{definition}</p>
    </div>
  </div>
);

interface CustomTag {
  [key: string]: string;
}

const NewTagDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (tag: CustomTag) => void;
  existingTags: string[];
}> = ({ isOpen, onClose, onSubmit, existingTags }) => {
  const [tagName, setTagName] = useState('');
  const [pattern, setPattern] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (existingTags.includes(tagName)) {
      setError('Tag name already exists');
      return;
    }

    if (!tagName || !pattern) {
      setError('Both fields are required');
      return;
    }

    onSubmit({ name: tagName, pattern });
    setTagName('');
    setPattern('');
    setError('');
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create Custom Tag</DialogTitle>
          <DialogDescription>
            Define a new custom tag with a regex pattern.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="tagName">Tag Name</Label>
              <Input
                id="tagName"
                value={tagName}
                onChange={(e) => setTagName(e.target.value.toUpperCase())}
                placeholder="CUSTOM_TAG_NAME"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="pattern">Regex Pattern</Label>
              <Input
                id="pattern"
                value={pattern}
                onChange={(e) => setPattern(e.target.value)}
                placeholder="\b[A-Z]{2}\d{6}\b"
              />
              <p className="text-sm text-gray-500">
                Example patterns:<br />
                Phone: \d{3}[-.]?\d{3}[-.]?\d{4}<br />
                Custom ID: [A-Z]{2}\d{6}
              </p>
            </div>
            {error && <p className="text-red-500 text-sm">{error}</p>}
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="default" className='bg-blue-400 hover:bg-blue-500'>Create Tag</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default function JobDetail() {
  const params = useParams();
  const reportId: string = params.jobId as string;
  const [lastUpdated, setLastUpdated] = useState(0);
  const [tabValue, setTabValue] = useState('configuration');
  const [selectedSource, setSelectedSource] = useState<'s3' | 'local'>('s3');

  // Remove selectedTags state, just keep availableTags
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [availableTagsCount, setAvailableTagsCount] = useState<{ type: string, count: number }[]>([]);

  const [reportData, setReportData] = useState<Report | null>(null);
  const [customTags, setCustomTags] = useState<CustomTag[]>([]);
  const [isNewTagDialogOpen, setIsNewTagDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  const fetchTags = async () => {
    setIsLoading(true);
    try {

      const report = await nerService.getReport(reportId);

      setReportData(report as Report);
      
      // Set selectedSource based on IsUpload field
      if (report.IsUpload) {
        setSelectedSource('local');
      } else {
        setSelectedSource('s3');
      }

      if (report.Tags) {
        const allTags: string[] = report.Tags;
        setAvailableTags(allTags);
      }
      if (report.CustomTags !== undefined) {
        const customTagsObj = report.CustomTags;
        const customTagName: string[] = Object.keys(customTagsObj);
        const allCustomTags: CustomTag[] = customTagName.map((tag) => ({
          name: tag,
          pattern: customTagsObj[tag]
        }));

        setCustomTags(allCustomTags);
      }

      if (report.TagCounts) {
        const tagObject = report.TagCounts;
        const tags = Object.keys(report.TagCounts);
        const allTagsCounts = tags.map((tag) => {
          return (
            {
              type: tag,
              count: tagObject[tag]
            }
          )
        })
        setAvailableTagsCount(allTagsCounts);
      }

    } catch (error) {
      console.error("Error fetching tags:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTags();

    // Set up refresh timer
    const timer = setInterval(() => {
      setLastUpdated(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [reportId]);

  // Optional: Add a refresh function that gets called when the refresh button is clicked
  const handleRefresh = () => {
    setLastUpdated(0);
    fetchTags();
  };

  // Define real data loading functions for entities
  const loadRealClassifiedTokenRecords = async (offset = 0, limit = 50) => {
    try {
      const entities = await nerService.getReportEntities(reportId, { offset, limit });
      console.log("API entities response:", entities.length > 0 ? entities[0] : "No entities");

      // Extract unique entity types from the API response
      const uniqueTypes = new Set(entities.map(entity => entity.Label));
      console.log("Unique entity types in API response:", Array.from(uniqueTypes));
      console.log("Tag filters available in UI:", availableTags);

      return entities.map(entity => {
        const record = {
          token: entity.Text,
          tag: entity.Label,
          sourceObject: entity.Object,
          context: {
            left: entity.LContext || '',
            right: entity.RContext || ''
          },
          start: entity.Start,
          end: entity.End,
          groups: reportData?.Groups?.filter(group => group.Objects?.includes(entity.Object)).map(g => g.Name) || []
        };

        // Log the first transformed record for debugging
        if (entity === entities[0]) {
          console.log("Transformed record:", record);
        }

        return record;
      });
    } catch (error) {
      console.error("Error loading entities:", error);
      return [];
    }
  };

  // Loads data for the Objects tab view, showing complete sentences with tagged tokens
  // Uses the GET /reports/{report_id}/entities endpoint and transforms the data
  // since the /objects endpoint doesn't exist yet in the backend
  const loadRealObjectRecords = async () => {
    try {
      // Get object previews (transformed from entities)
      const objectPreviews = await nerService.getReportObjects(reportId, { limit: 100 });

      return objectPreviews.map(preview => {
        // Create tagged tokens array from the parallel tokens and tags arrays
        const taggedTokens = preview.tokens.map((token, index) => [
          token,
          preview.tags[index]
        ]) as [string, string][];

        return {
          taggedTokens,
          sourceObject: preview.object,
          // Get groups that include this object
          groups: reportData?.Groups?.filter(group =>
            group.Objects?.includes(preview.object)
          ).map(g => g.Name) || []
        };
      });
    } catch (error) {
      console.error("Error loading object records:", error);
      return [];
    }
  };

  return (
    <div className="container px-4 py-8 mx-auto">
      {/* Breadcrumbs */}
      <div className="mb-6">
        <div className="flex items-center mb-2">
          <Link href={`/token-classification/${params.deploymentId}?tab=jobs`} className="text-blue-500 hover:underline">
            Jobs
          </Link>
          <span className="mx-2 text-gray-400">/</span>
          <span className="text-gray-700">{reportData?.ReportName || '[Job Name]'}</span>
        </div>
      </div>

      {/* Title and Back Button */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-medium">{reportData?.ReportName || '[Job Name]'}</h1>

        <Button variant="outline" size="sm" asChild>
          <Link href={`/token-classification/${params.deploymentId}?tab=jobs`} className="flex items-center">
            <ArrowLeft className="mr-1 h-4 w-4" /> Back to Jobs
          </Link>
        </Button>
      </div>

      {/* Tabs and Controls */}
      <Tabs value={tabValue} onValueChange={setTabValue} className="w-full">
        <div className="flex items-center justify-between border-b mb-6">
          <TabsList className="border-0 bg-transparent p-0">
            <TabsTrigger
              value="configuration"
              className="data-[state=active]:border-b-2 data-[state=active]:border-blue-500 data-[state=active]:shadow-none rounded-none bg-transparent px-4 py-3 data-[state=active]:bg-transparent"
            >
              Configuration
            </TabsTrigger>
            <TabsTrigger
              value="analytics"
              className="data-[state=active]:border-b-2 data-[state=active]:border-blue-500 data-[state=active]:shadow-none rounded-none bg-transparent px-4 py-3 data-[state=active]:bg-transparent"
            >
              Analytics
            </TabsTrigger>
            <TabsTrigger
              value="output"
              className="data-[state=active]:border-b-2 data-[state=active]:border-blue-500 data-[state=active]:shadow-none rounded-none bg-transparent px-4 py-3 data-[state=active]:bg-transparent"
            >
              Output
            </TabsTrigger>
          </TabsList>

          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-500">Last updated: {lastUpdated} seconds ago</span>
            <Button variant="ghost" size="icon" onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon">
              <Pause className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon">
              <Square className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <TabsContent value="configuration" className="mt-0">
          <div className="space-y-8">
            {/* Source section */}
            <div>
              <h2 className="text-lg font-medium mb-4">Source</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {(reportData?.SourceS3Bucket) && <SourceOption
                  title="S3 Bucket"
                  description={(reportData.SourceS3Bucket + "/" + reportData?.SourceS3Prefix) || "s3://thirdai-dev/customer-calls/2025/"}
                  isSelected={selectedSource === 's3'}
                  disabled={selectedSource === 'local'}
                  onClick={() => { }}
                />}
                <SourceOption
                  title="File Upload"
                  description=""
                  isSelected={selectedSource === 'local'}
                  disabled={selectedSource === "s3"}
                  onClick={() => { }}
                />
              </div>
            </div>

            {/* Tags section */}
            <div>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-medium">Tags</h2>
              </div>

              {isLoading ? (
                <div className="flex justify-center py-4">
                  <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : availableTags.length === 0 ? (
                <div className="text-gray-500 py-2">No tags available</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {availableTags.map(tag => (
                    <Tag
                      key={tag}
                      tag={tag}
                      displayOnly={true}
                    />
                  ))}
                </div>
              )}
            </div>

            {/* Custom Tags section */}
            {reportData?.CustomTags && <div>
              <h2 className="text-lg font-medium mb-4">Custom Tags</h2>
              {isLoading ? (
                <div className="flex justify-center py-4">
                  <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {customTags.map((customTag) => (
                    <div key={customTag.name} className="border border-gray-200 rounded-md overflow-hidden">
                      <div className="p-4 border-b border-gray-200 flex justify-between items-center">
                        <Tag tag={customTag.name} custom displayOnly={true} />
                      </div>
                      <div className="p-4">
                        <p className="text-sm font-mono">{customTag.pattern}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>}

            {/* Groups section */}
            {reportData?.Groups && <div>
              <h2 className="text-lg font-medium mb-4">Groups</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {reportData.Groups.map((group) => {
                  return (
                    <GroupCard key={group.Id} name={group.Name} definition={group.Query} />
                  )
                })}

                <div
                  className="border border-dashed border-gray-300 rounded-md flex items-center justify-center p-6 cursor-pointer hover:border-gray-400"
                  onClick={() => console.log('Add new group')}
                >
                  <div className="flex flex-col items-center">
                    <Plus className="h-8 w-8 text-gray-400 mb-2" />
                    <span className="text-gray-600">Define new group</span>
                  </div>
                </div>
              </div>
            </div>}
          </div>
        </TabsContent>

        <TabsContent value="analytics">
          <AnalyticsDashboard
            progress={calculateProgress(reportData)}
            tokensProcessed={getProcessedTokens(reportData)}
            tags={availableTagsCount}
          />
        </TabsContent>

        <TabsContent value="output">
          <DatabaseTable
            loadMoreObjectRecords={loadRealObjectRecords}
            loadMoreClassifiedTokenRecords={loadRealClassifiedTokenRecords}
            groups={reportData?.Groups?.map(g => g.Name) || mockGroups}
            tags={availableTags}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
} 