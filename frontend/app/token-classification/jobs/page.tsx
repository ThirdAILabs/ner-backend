'use client';

import React, { useEffect } from 'react';
import { useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { ArrowLeft, RefreshCw, Square } from 'lucide-react';
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
} from '@/components/ui/dialog';
import { Box } from '@mui/material';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Suspense } from 'react';
import { floor } from 'lodash';

// Calculate progress based on InferenceTaskStatuses
const calculateProgress = (report: Report | null): number => {
  const successfulFiles = report?.SucceededFileCount || 0;
  const failedFiles = report?.FailedFileCount || 0;
  const totalFiles = report?.FileCount || 1;

  return floor(((successfulFiles + failedFiles) / totalFiles) * 100);
};

// Get the total number of processed tokens
const getProcessedTokens = (report: Report | null): number => {
  if (!report || !report.InferenceTaskStatuses) {
    return 0;
  }

  return (
    (report.InferenceTaskStatuses.COMPLETED?.TotalSize || 0) +
    (report.InferenceTaskStatuses.FAILED?.TotalSize || 0) +
    (report.InferenceTaskStatuses.RUNNING?.TotalSize || 0)
  );
};

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
  onClick,
}) => (
  <div
    className={`relative p-6 border rounded-md transition-all
      ${isSelected ? 'border-blue-500 border-2' : 'border-gray-200'}
      ${
        disabled
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

const Tag: React.FC<TagProps> = ({
  tag,
  selected = true,
  onClick,
  custom = false,
  addNew = false,
  displayOnly = false,
}) => {
  return (
    <div
      className={`px-3 py-1 text-sm font-medium rounded-sm ${displayOnly ? 'bg-blue-100 text-blue-800' : selected ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'} ${!displayOnly && onClick ? 'cursor-pointer' : ''}`}
      style={{ userSelect: 'none' }}
      onClick={displayOnly ? undefined : onClick}
    >
      {tag}
    </div>
  );
};

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
          <DialogDescription>Define a new custom tag with a regex pattern.</DialogDescription>
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
                Example patterns:
                <br />
                Phone: \d{3}[-.]?\d{3}[-.]?\d{4}
                <br />
                Custom ID: [A-Z]{2}\d{6}
              </p>
            </div>
            {error && <p className="text-red-500 text-sm">{error}</p>}
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" variant="default" className="bg-blue-400 hover:bg-blue-500">
              Create Tag
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

function JobDetail() {
  const searchParams = useSearchParams();
  const reportId: string = searchParams.get('jobId') as string;
  const [tabValue, setTabValue] = useState('analytics');
  const [selectedSource, setSelectedSource] = useState<'s3' | 'local'>('s3');
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  // Remove selectedTags state, just keep availableTags
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [availableTagsCount, setAvailableTagsCount] = useState<{ type: string; count: number }[]>(
    []
  );

  const [timeTaken, setTimeTaken] = useState(0);

  const [reportData, setReportData] = useState<Report | null>(null);
  const [customTags, setCustomTags] = useState<CustomTag[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [dataProcessed, setDataProcessed] = useState<number | null>(null);

  const tabChangeByGraph = React.useRef(false);

  function setDataProcessedFromReport(report: Report | null) {
    if (report) {
      setDataProcessed(
        (report.InferenceTaskStatuses?.COMPLETED?.CompletedSize || 0) +
          (report.InferenceTaskStatuses?.FAILED?.CompletedSize || 0) +
          (report.InferenceTaskStatuses?.RUNNING?.CompletedSize || 0)
      );
    }
  }

  const fetchTags = async () => {
    setIsLoading(true);
    try {
      const report = await nerService.getReport(reportId);

      setReportData(report as Report);

      setDataProcessedFromReport(reportData);

      setTimeTaken((report.TotalInferenceTimeSeconds || 0) + (report.ShardDataTimeSeconds || 0));

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
          pattern: customTagsObj[tag],
        }));

        setCustomTags(allCustomTags);
      }

      if (report.TagCounts) {
        const tagObject = report.TagCounts;
        const tags = Object.keys(report.TagCounts);
        const allTagsCounts = tags.map((tag) => {
          return {
            type: tag,
            count: tagObject[tag],
          };
        });
        setAvailableTagsCount(allTagsCounts);
      }
    } catch (error) {
      console.error('Error fetching tags:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    let pollInterval: NodeJS.Timeout;

    const poll = async () => {
      await fetchTags();
      const currentProgress = calculateProgress(reportData);

      if (currentProgress === 100) {
        clearInterval(pollInterval);
      }
    };

    poll();
    pollInterval = setInterval(poll, 5000);

    return () => {
      clearInterval(pollInterval);
    };
  }, [reportId, reportData?.SucceededFileCount]);

  useEffect(() => {
    // There are two ways to get to the 'Review' tab:
    // 1. By clicking on the tab, in which case we want to have all the filters selected.
    // 2. By clicking on a bar in the graph, in which case we select that label as the selected tag.
    // This code is needed to reset the filters selected when the user clicks on the tab directly.
    if (tabValue === 'output') {
      if (!tabChangeByGraph.current) {
        setSelectedTag(null);
      } else {
        tabChangeByGraph.current = false;
      }
    }
  }, [tabValue]);

  return (
    <div className="container px-4 py-8 mx-auto" style={{ width: '90%' }}>
      {/* Header with Back Button and Title */}
      <div className="flex items-center justify-between mb-6">
        <Button variant="outline" size="sm" asChild>
          <Link href={`/token-classification/landing?tab=jobs`} className="flex items-center">
            <ArrowLeft className="mr-1 h-4 w-4" /> Back to Scans
          </Link>
        </Button>
        <h1 className="text-2xl font-medium text-center flex-1">
          {reportData?.ReportName || '[Scan Name]'}
        </h1>
        {/* Empty div to maintain spacing */}
        <div className="w-[106px]"></div> {/* Width matches the Back button */}
      </div>

      {/* Tabs and Controls */}
      <Tabs value={tabValue} onValueChange={setTabValue}>
        <div className="flex items-center justify-between border-b mb-6">
          <TabsList className="border-0 bg-transparent p-0">
            <TabsTrigger
              value="analytics"
              className="data-[state=active]:border-b-2 data-[state=active]:border-blue-500 data-[state=active]:shadow-none rounded-none bg-transparent px-4 py-3 data-[state=active]:bg-transparent"
            >
              Summary
            </TabsTrigger>
            <TabsTrigger
              value="output"
              className="data-[state=active]:border-b-2 data-[state=active]:border-blue-500 data-[state=active]:shadow-none rounded-none bg-transparent px-4 py-3 data-[state=active]:bg-transparent"
            >
              Review
            </TabsTrigger>
            <TabsTrigger
              value="configuration"
              className="data-[state=active]:border-b-2 data-[state=active]:border-blue-500 data-[state=active]:shadow-none rounded-none bg-transparent px-4 py-3 data-[state=active]:bg-transparent"
            >
              Info
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="configuration" className="mt-0">
          {/* STARTS */}
          {/* Source */}
          <Box className="bg-muted/60" sx={{ p: 3, borderRadius: 3 }}>
            <h2 className="text-2xl font-medium mb-4">Source</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {selectedSource === 's3' && reportData?.SourceS3Bucket && (
                <Box
                  sx={{
                    p: 2,
                    bgcolor: 'grey.50',
                    borderRadius: 2,
                    boxShadow: 1,
                  }}
                >
                  <h3 className="text-lg font-medium mb-1">S3 Bucket</h3>
                  {reportData.S3Endpoint && (
                    <p className="text-sm text-gray-600">
                      <b>Endpoint:</b> {reportData.S3Endpoint}
                    </p>
                  )}
                  {reportData.S3Region && (
                    <p className="text-sm text-gray-600">
                      <b>Region:</b> {reportData.S3Region}
                    </p>
                  )}
                  <p className="text-sm text-gray-600">
                    <b>Bucket:</b> {reportData.SourceS3Bucket}
                  </p>
                  {reportData.SourceS3Prefix && (
                    <p className="text-sm text-gray-600">
                      <b>Prefix:</b> {reportData.SourceS3Prefix}
                    </p>
                  )}
                </Box>
              )}

              {selectedSource === 'local' && (
                <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
                  <h3 className="text-lg font-medium mb-1">Local Files</h3>
                  {/* <p className="text-sm text-gray-600">File Location...</p> */}
                </Box>
              )}
            </div>
          </Box>

          {/* Tags */}
          <Box className="bg-muted/60" sx={{ p: 3, borderRadius: 3, marginTop: 3 }}>
            <h2 className="text-2xl font-medium mb-4">Tags</h2>
            <div className="flex justify-between items-center mb-4">
              {isLoading ? (
                <div className="flex justify-center py-4">
                  <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : availableTags.length === 0 ? (
                <div className="text-gray-500 py-2">No tags available</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {availableTags.map((tag) => (
                    <Tag key={tag} tag={tag} displayOnly={true} />
                  ))}
                </div>
              )}
            </div>
          </Box>

          {/* Custom Tags */}
          <Box className="bg-muted/60" sx={{ p: 3, borderRadius: 3, marginTop: 3 }}>
            <h2 className="text-2xl font-medium mb-4">Tags</h2>
            <div className="flex justify-between items-center mb-4">
              {isLoading ? (
                <div className="flex justify-center py-4">
                  <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : customTags?.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {customTags.map((customTag) => (
                    <div
                      key={customTag.name}
                      className="border border-gray-300 rounded-md overflow-hidden"
                    >
                      <div className="p-4 border-b border-gray-300  flex justify-between items-center">
                        <Tag tag={customTag.name} custom displayOnly={true} />
                      </div>
                      <div className="p-4">
                        <p className="text-sm font-mono">{customTag.pattern}</p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-10 bg-gray-50 border border-dashed border-gray-200 rounded-lg w-[400px]">
                  <p className="text-gray-500">No custom tags defined for this report</p>
                </div>
              )}
            </div>
          </Box>

          {/* Groups */}
          <Box className="bg-muted/60" sx={{ p: 3, borderRadius: 3, marginTop: 3 }}>
            <h2 className="text-2xl font-medium mb-4">Groups</h2>
            <div className="flex justify-between items-center mb-4">
              {isLoading ? (
                <div className="flex justify-center py-4">
                  <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : (reportData?.Groups ?? []).length > 0 ? (
                <div className="grid grid-cols-3 md:grid-cols-3 gap-4">
                  {reportData?.Groups?.map((group) => (
                    <GroupCard key={group.Id} name={group.Name} definition={group.Query} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-10 bg-gray-50 border border-dashed border-gray-200 rounded-lg w-[400px]">
                  <p className="text-gray-500">No groups defined for this report</p>
                </div>
              )}
            </div>
          </Box>

          {/* ENDS */}
        </TabsContent>

        <TabsContent value="analytics">
          <AnalyticsDashboard
            tokensProcessed={getProcessedTokens(reportData)}
            tags={availableTagsCount}
            timeTaken={timeTaken}
            succeededFileCount={reportData?.SucceededFileCount || 0}
            failedFileCount={reportData?.FailedFileCount || 0}
            totalFileCount={reportData?.FileCount || 1}
            dataProcessed={dataProcessed || 0}
            setTab={(val) => {
              tabChangeByGraph.current = true;
              setTabValue(val);
            }}
            setSelectedTag={(tag) => {
              setSelectedTag(tag);
            }}
          />
        </TabsContent>

        <TabsContent value="output">
          <DatabaseTable
            groups={reportData?.Groups?.map((g) => g.Name) || []}
            tags={availableTagsCount}
            uploadId={reportData?.IsUpload ? reportData?.SourceS3Prefix : ''}
            initialSelectedTag={selectedTag}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default function Page() {
  return (
    <Suspense>
      <JobDetail />
    </Suspense>
  );
}
