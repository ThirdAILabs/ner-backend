'use client';

import React, { useEffect } from 'react';
import { useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { CheckCircle, ArrowLeft, RefreshCw, Pause, Square, Plus, Edit } from 'lucide-react';
import { AnalyticsDashboard } from '@/components/AnalyticsDashboard';
import { DatabaseTable } from './(database-table)/DatabaseTable';
import { nerService } from '@/lib/backend';
import { usePathname } from 'next/navigation';

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
const mockTags = ['VIN', 'NAME', 'ORG', 'ADDRESS', 'EMAIL', 'SSN', 'PHONE', 'POLICY_ID', 'MED_REC_NO', 'LICENSE', 'EMPLOYER', 'ID', 'USERNAME', 'URL', 'IP_ADDR', 'ZIP_CODE', 'ACCOUNT', 'INS_PROV', 'PROCEDURE', 'DATE', 'NATIONALITY', 'SERIAL_NO', 'CRED_CARD_NUM', 'CVV'];

const loadMoreMockObjectRecords = () => {
  return Promise.resolve([
    {
      taggedTokens: [
        ['My', 'O'] as [string, string],
        ['name', 'O'] as [string, string],
        ['is', 'O'] as [string, string],
        ['John', 'NAME'] as [string, string],
        ['Smith', 'NAME'] as [string, string],
        ['and', 'O'] as [string, string],
        ['my', 'O'] as [string, string],
        ['social', 'O'] as [string, string],
        ['is', 'O'] as [string, string],
        ['123-45-6789', 'SSN'] as [string, string],
      ],
      sourceObject: 'call_transcript_1.txt',
      groups: ['Sensitive'],
    },
    {
      taggedTokens: [
        ['Jane', 'NAME'] as [string, string],
        ['Doe', 'NAME'] as [string, string],
        ['at', 'O'] as [string, string],
        ['123', 'ADDRESS'] as [string, string],
        ['Main', 'ADDRESS'] as [string, string],
        ['St', 'ADDRESS'] as [string, string],
        ['with', 'O'] as [string, string],
        ['vehicle', 'O'] as [string, string],
        ['1HGCM82633A004352', 'VIN'] as [string, string],
      ],
      sourceObject: 'call_transcript_2.txt',
      groups: ['Reject'],
    },
  ]);
};

const loadMoreMockClassifiedTokenRecords = () => {
  return Promise.resolve([
    {
      token: 'John',
      tag: 'NAME',
      sourceObject: 'call_transcript_1.txt',
      groups: ['Sensitive'],
    },
    {
      token: 'Smith',
      tag: 'NAME',
      sourceObject: 'call_transcript_1.txt',
      groups: ['Sensitive'],
    },
    {
      token: '123-45-6789',
      tag: 'SSN',
      sourceObject: 'call_transcript_1.txt',
      groups: ['Sensitive'],
    },
    {
      token: 'Jane',
      tag: 'NAME',
      sourceObject: 'call_transcript_2.txt',
      groups: ['Reject'],
    },
    {
      token: 'Doe',
      tag: 'NAME',
      sourceObject: 'call_transcript_2.txt',
      groups: ['Reject'],
    },
    {
      token: '123 Main St',
      tag: 'ADDRESS',
      sourceObject: 'call_transcript_2.txt',
      groups: ['Reject'],
    },
    {
      token: '1HGCM82633A004352',
      tag: 'VIN',
      sourceObject: 'call_transcript_2.txt',
      groups: ['Reject'],
    },
  ]);
};

// Source option card component
interface SourceOptionProps {
  title: string;
  description: string;
  isSelected?: boolean;
  onClick: () => void;
}

const SourceOption: React.FC<SourceOptionProps> = ({ title, description, isSelected = false, onClick }) => (
  <div
    className={`relative p-6 border rounded-md cursor-pointer transition-all
      ${isSelected ? 'border-blue-500 border-2' : 'border-gray-200 hover:border-blue-300'}
    `}
    onClick={onClick}
  >
    <h3 className="text-base font-medium">{title}</h3>
    <p className="text-sm text-gray-500 mt-1">{description}</p>

    {isSelected && (
      <div className="absolute top-3 right-3">
        <Edit className="h-4 w-4 text-gray-500" />
      </div>
    )}
  </div>
);

// Tag chip component
interface TagProps {
  tag: string;
  selected?: boolean;
  onClick?: () => void;
}

const Tag: React.FC<TagProps> = ({ tag, selected = true, onClick }) => (
  <div
    className={`px-3 py-1.5 text-sm font-medium rounded-sm cursor-pointer
      ${selected ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}
    `}
    style={{ userSelect: 'none' }}
    onClick={onClick}
  >
    {tag}
  </div>
);

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

export default function JobDetail() {
  const params = useParams();
  const reportId: string = params.jobId as string;
  const [lastUpdated, setLastUpdated] = useState(0);
  const [tabValue, setTabValue] = useState('configuration');
  const [selectedSource, setSelectedSource] = useState('s3');
  const [selectedSaveLocation, setSelectedSaveLocation] = useState('s3');
  const [selectedTags, setSelectedTags] = useState<string[]>(mockTags);
  const [dynamicTags, setDynamicTags] = useState<string[]>(mockTags);

  const [reportData, setReportData] = useState<Report | null>(null);

  // Gather unique entity types from API
  const fetchAndProcessEntities = async () => {
    try {
      const entities = await nerService.getReportEntities(reportId, { limit: 200 });
      if (entities && entities.length > 0) {
        // Extract and deduplicate tag types
        const apiTagTypes = Array.from(new Set(entities.map(e => e.Label)));

        // Combine with mockTags (for backward compatibility) and deduplicate
        const combinedTags = Array.from(new Set([...mockTags, ...apiTagTypes]));
        console.log("Combined tag types:", combinedTags);

        setDynamicTags(combinedTags);
      }
    } catch (error) {
      console.error("Error fetching entity types:", error);
    }
  };

  const fetchReportData = async () => {
    const response = await nerService.getReport(reportId);
    console.log(response);
    setReportData(response);
  }

  useEffect(() => {
    fetchReportData();
    fetchAndProcessEntities();
  }, []);

  // Define real data loading functions for entities
  const loadRealClassifiedTokenRecords = async (offset = 0, limit = 50) => {
    try {
      const entities = await nerService.getReportEntities(reportId, { offset, limit });
      console.log("API entities response:", entities.length > 0 ? entities[0] : "No entities");

      // Extract unique entity types from the API response
      const uniqueTypes = new Set(entities.map(entity => entity.Label));
      console.log("Unique entity types in API response:", Array.from(uniqueTypes));
      console.log("Tag filters available in UI:", mockTags);

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

  // This is a placeholder until we have an endpoint for full object data
  const loadRealObjectRecords = async () => {
    try {
      // Group entities by object
      const entities = await nerService.getReportEntities(reportId, { limit: 200 });
      const objectMap = new Map<string, [string, string, string, string][]>();

      entities.forEach(entity => {
        if (!objectMap.has(entity.Object)) {
          objectMap.set(entity.Object, []);
        }
        // Store [text, label, left context, right context]
        objectMap.get(entity.Object)?.push([
          entity.Text,
          entity.Label,
          entity.LContext || '',
          entity.RContext || ''
        ]);
      });

      return Array.from(objectMap.entries()).map(([objectName, tokens]) => ({
        taggedTokens: tokens.map(t => [t[0], t[1]]) as [string, string][],
        tokenContexts: tokens.map(t => ({ left: t[2], right: t[3] })),
        sourceObject: objectName,
        groups: reportData?.Groups?.filter(group => group.Objects?.includes(objectName)).map(g => g.Name) || []
      }));
    } catch (error) {
      console.error("Error loading object records:", error);
      return [];
    }
  };

  const toggleTag = (tag: string) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter(t => t !== tag));
    } else {
      setSelectedTags([...selectedTags, tag]);
    }
  };

  const selectAllTags = () => {
    setSelectedTags(mockTags);
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
          <span className="text-gray-700">Customer Calls</span>
        </div>
      </div>

      {/* Title and Back Button */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-medium">Customer Calls</h1>

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
            <Button variant="ghost" size="icon" onClick={() => setLastUpdated(0)}>
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
                  onClick={() => { }}
                />}
                <SourceOption
                  title="File Upload"
                  description="Configure now"
                  isSelected={selectedSource === 'local'}
                  onClick={() => { }}
                />
              </div>
            </div>

            {/* Tags section */}
            <div>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-medium">Tags</h2>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={selectAllTags}
                  className="text-sm flex items-center"
                  disabled={selectedTags?.length === mockTags?.length}
                >
                  <span className="mr-1">Select All</span>
                  <input
                    type="checkbox"
                    checked={selectedTags.length === mockTags.length}
                    onChange={selectAllTags}
                    className="rounded border-gray-300"
                  />
                </Button>
              </div>
              <div className="flex flex-wrap gap-2">
                {mockTags.map(tag => (
                  <Tag
                    key={tag}
                    tag={tag}
                    selected={selectedTags.includes(tag)}
                    onClick={() => toggleTag(tag)}
                  />
                ))}
              </div>
            </div>

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

            {/* Save Groups To section */}
            <div>
              <h2 className="text-lg font-medium mb-4">Save Groups To</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {reportData?.SourceS3Bucket && <SourceOption
                  title="S3 Bucket"
                  description={(reportData.SourceS3Bucket + "/" + reportData?.SourceS3Prefix) || "s3://thirdai-dev/customer-calls/2025/"}
                  isSelected={selectedSaveLocation === 's3'}
                  onClick={() => setSelectedSaveLocation('s3')}
                />}
                <SourceOption
                  title="Local Storage"
                  description="local"
                  isSelected={selectedSaveLocation === 'local'}
                  onClick={() => setSelectedSaveLocation('local')}
                />
                <SourceOption
                  title="No storage location"
                  description="You can still save groups"
                  isSelected={selectedSaveLocation === 'none'}
                  onClick={() => setSelectedSaveLocation('none')}
                />
                <SourceOption
                  title="More options"
                  description="coming soon"
                  isSelected={selectedSaveLocation === 'more'}
                  onClick={() => setSelectedSaveLocation('more')}
                />
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="analytics">
          <AnalyticsDashboard
            progress={calculateProgress(reportData)}
            tokensProcessed={getProcessedTokens(reportData)}
            tokenCounts={{
              'VIN': 450,
              'NAME': 2300,
              'SSN': 800,
              'ADDRESS': 1200,
              'EMAIL': 950,
              'PHONE': 1100,
              'POLICY_ID': 400,
              'DATE': 780
            }}
          />
        </TabsContent>

        <TabsContent value="output">
          <DatabaseTable
            loadMoreObjectRecords={loadRealObjectRecords}
            loadMoreClassifiedTokenRecords={loadRealClassifiedTokenRecords}
            groups={reportData?.Groups?.map(g => g.Name) || mockGroups}
            tags={dynamicTags}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
} 