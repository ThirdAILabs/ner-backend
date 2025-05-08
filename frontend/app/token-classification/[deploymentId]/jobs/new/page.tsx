'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Plus, RefreshCw, Edit } from 'lucide-react';
import { nerService } from '@/lib/backend';

// Tag chip component - reused from the detail page but with interactive mode
interface TagProps {
  tag: string;
  selected?: boolean;
  onClick?: () => void;
  custom?: boolean;
  addNew?: boolean;
}

const Tag: React.FC<TagProps> = ({ tag, selected = false, onClick, custom = false, addNew = false }) => {
  return (
    <div
      className={`px-3 py-1 text-sm font-medium rounded-sm ${!custom && "cursor-pointer"} ${selected ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-700 hover:bg-gray-200"}`}
      style={{ userSelect: 'none' }}
      onClick={onClick}
    >
      {tag}
    </div>
  );
}

// Source option card component - reused from the detail page
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

// Group card component
interface GroupProps {
  name: string;
  definition: string;
  onRemove: () => void;
}

const GroupCard: React.FC<GroupProps> = ({ name, definition, onRemove }) => (
  <div className="border border-gray-200 rounded-md overflow-hidden">
    <div className="p-4 border-b border-gray-200 flex justify-between items-center">
      <h3 className="text-base font-medium">{name}</h3>
      <Button variant="ghost" size="sm" onClick={onRemove} className="text-red-500">
        Remove
      </Button>
    </div>
    <div className="p-4">
      <p className="text-sm font-mono">{definition}</p>
    </div>
  </div>
);

// Custom Tag interface
interface CustomTag {
  name: string;
  pattern: string;
}

export default function NewJobPage() {
  const params = useParams();
  const router = useRouter();
  const deploymentId = params.deploymentId as string;

  // Essential state
  const [selectedSource, setSelectedSource] = useState<'s3' | 'files'>('s3');
  const [sourceS3Bucket, setSourceS3Bucket] = useState('');
  const [sourceS3Prefix, setSourceS3Prefix] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  //Job Name
  const [jobName, setJobName] = useState('');

  // Model selection
  const [models, setModels] = useState<any[]>([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [selectedModel, setSelectedModel] = useState<any>(null);

  // Tags handling
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [isTagsLoading, setIsTagsLoading] = useState(false);

  // Groups handling
  const [groupName, setGroupName] = useState('');
  const [groupQuery, setGroupQuery] = useState('');
  const [groups, setGroups] = useState<Record<string, string>>({});

  // Custom tags handling
  const [customTags, setCustomTags] = useState<CustomTag[]>([]);
  const [customTagName, setCustomTagName] = useState('');
  const [customTagPattern, setCustomTagPattern] = useState('');
  const [isCustomTagDialogOpen, setIsCustomTagDialogOpen] = useState(false);
  const [editingTag, setEditingTag] = useState<CustomTag | null>(null);

  // Error/Success messages
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const [patternType, setPatternType] = useState('string');

  // Fetch models on page load
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const modelData = await nerService.listModels();
        // Only show trained models that can be used for inference
        const trainedModels = modelData.filter(model => model.Status === 'TRAINED');
        setModels(trainedModels);
      } catch (err) {
        console.error('Error fetching models:', err);
        setError('Failed to load models. Please try again.');
      }
    };

    fetchModels();
  }, []);

  // Load tags when a model is selected
  useEffect(() => {
    if (!selectedModelId) return;

    const fetchTags = async () => {
      setIsTagsLoading(true);
      try {
        const model = await nerService.getModel(selectedModelId);
        setSelectedModel(model);

        // Get tags from the model
        const modelTags = model.Tags || [];
        console.log("Tags from model:", modelTags);

        setAvailableTags(modelTags);
        setSelectedTags(modelTags); // By default, select all tags

      } catch (error) {
        console.error("Error fetching tags:", error);
        setError("Failed to load tags from the selected model");
      } finally {
        setIsTagsLoading(false);
      }
    };

    fetchTags();
  }, [selectedModelId]);

  // Toggle tag selection
  const toggleTag = (tag: string) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter(t => t !== tag));
    } else {
      setSelectedTags([...selectedTags, tag]);
    }
  };

  // Select all tags
  const selectAllTags = () => {
    setSelectedTags([...availableTags]);
  };

  // Add a new group
  const handleAddGroup = () => {
    if (!groupName.trim() || !groupQuery.trim()) {
      setError('Group name and query are required');
      return;
    }

    setGroups({
      ...groups,
      [groupName]: groupQuery
    });

    setGroupName('');
    setGroupQuery('');
    setError(null);
  };

  // Remove a group
  const handleRemoveGroup = (name: string) => {
    const newGroups = { ...groups };
    delete newGroups[name];
    setGroups(newGroups);
  };

  // Add a custom tag
  const handleAddCustomTag = () => {    
    if (!customTagName.trim() || !customTagPattern.trim()) {
      setError('Custom tag name and pattern are required');
      return;
    }

    if (editingTag && customTagName.trim().toUpperCase() === editingTag.name.toUpperCase()) {
      const newCustomTag = {
        name: customTagName.trim().toUpperCase(),
        pattern: customTagPattern
      }

      setCustomTags(prev => prev.map(tag =>
        tag.name === editingTag.name ? newCustomTag : tag
      ));
    } else {
      for (let index = 0; index < customTags.length; index++) {
        const thisTag = customTags[index];
        if (thisTag.name === customTagName.toUpperCase()) {
          setError('Custom Tag name must be unique');
          return;
        }
      }
      setCustomTags(prev => [...prev, {
        name: customTagName.trim().toUpperCase(),
        pattern: customTagPattern
      }]);
    }

    setCustomTagName('');
    setCustomTagPattern('');
    setEditingTag(null);
    setIsCustomTagDialogOpen(false);
  };
  const handleRemoveCustomTag = (tagName: string) => {
    setCustomTags(prev => prev.filter(tag => tag.name !== tagName));
    setAvailableTags(prev => prev.filter(tag => tag !== tagName));
    setSelectedTags(prev => prev.filter(tag => tag !== tagName));
  };
  const handleEditCustomTag = (tag: CustomTag) => {
    setCustomTagName(tag.name);
    setCustomTagPattern(tag.pattern);
    setEditingTag(tag);
    setIsCustomTagDialogOpen(true);

  };

  const areFilesIdentical = (file1: File, file2: File): boolean => {
    return file1.name === file2.name && file1.size === file2.size;
  };

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      const fileArray = Array.from(files);

      // Filter out duplicates
      const newFiles = fileArray.filter(newFile => {
        // Check if this file already exists in selectedFiles
        const isDuplicate = selectedFiles.some(existingFile =>
          areFilesIdentical(existingFile, newFile)
        );
        return !isDuplicate;
      });

      setSelectedFiles(prev => [...prev, ...newFiles]);
      e.target.value = '';
    }
  };

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };
  // Submit the new job
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedModelId) {
      setError('Please select a model');
      return;
    }

    if (selectedSource === 's3' && !sourceS3Bucket) {
      setError('S3 bucket is required');
      return;
    }

    if (selectedSource === 'files' && selectedFiles.length === 0) {
      setError('Please select at least one file');
      return;
    }

    if (selectedTags.length === 0) {
      setError('Please select at least one tag');
      return;
    }

    setError(null);
    setIsSubmitting(true);

    try {
      let uploadId;

      // Handle file uploads if needed
      if (selectedSource === 'files') {
        const uploadResponse = await nerService.uploadFiles(selectedFiles);
        uploadId = uploadResponse.Id;
      }

      // Create custom tags object for API
      const customTagsObj: Record<string, string> = {};
      customTags.forEach(tag => {
        customTagsObj[tag.name] = tag.pattern;
      });

      // Create the report
      console.log("Job Name: ", jobName);
      const response = await nerService.createReport({
        ModelId: selectedModelId,
        Tags: selectedTags,
        CustomTags: customTagsObj,
        ...(selectedSource === 's3' ? {
          SourceS3Bucket: sourceS3Bucket,
          SourceS3Prefix: sourceS3Prefix || undefined,
        } : {
          UploadId: uploadId,
        }),
        Groups: groups,
        report_name: jobName
      });

      setSuccess(true);

      // Redirect after success
      setTimeout(() => {
        router.push(`/token-classification/${deploymentId}/jobs/${response.ReportId}`);
      }, 2000);
    } catch (err) {
      setError('Failed to create report. Please try again.');
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container px-4 py-8 mx-auto">
      {/* Breadcrumbs */}
      <div className="mb-6">
        <div className="flex items-center mb-2">
          <Link href={`/token-classification/${deploymentId}?tab=jobs`} className="text-blue-500 hover:underline">
            Jobs
          </Link>
          <span className="mx-2 text-gray-400">/</span>
          <span className="text-gray-700">New Job</span>
        </div>
      </div>

      {/* Title and Back Button */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-medium">Create New Job</h1>

        <Button variant="outline" size="sm" asChild>
          <Link href={`/token-classification/${deploymentId}?tab=jobs`} className="flex items-center">
            <ArrowLeft className="mr-1 h-4 w-4" /> Back to Jobs
          </Link>
        </Button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
          {error}
        </div>
      )}

      {success ? (
        <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded mb-6">
          Job created successfully! Redirecting...
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Job Name Field */}
          <div>
            <h2 className="text-lg font-medium mb-4">Job Name</h2>
            <div className="w-full md:w-1/2">
              <input
                type="text"
                value={jobName}
                onChange={(e) => {
                  const value = e.target.value.replace(/\s/g, '_');
                  setJobName(value);
                }}
                className="w-full p-2 border border-gray-300 rounded"
                placeholder="my_job_name"
                required
                pattern="^[^\s]+$"
              />
              <p className="text-sm text-gray-500 mt-1">
                Use only letters, numbers, and underscores. No spaces allowed.
              </p>
            </div>
          </div>

          {/* Model Selection */}
          <div>
            <h2 className="text-lg font-medium mb-4">Model</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {models.map(model => (
                <SourceOption
                  key={model.Id}
                  title={model.Name}
                  description={`Description: TBD`}
                  // description={`Type: ${model.Type}`}
                  isSelected={selectedModelId === model.Id}
                  onClick={() => setSelectedModelId(model.Id)}
                />
              ))}
            </div>
          </div>

          {/* Source Section */}
          <div>
            <h2 className="text-lg font-medium mb-4">Source</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <SourceOption
                title="S3 Bucket"
                description="Use files from an S3 bucket"
                isSelected={selectedSource === 's3'}
                onClick={() => setSelectedSource('s3')}
              />
              <SourceOption
                title="File Upload"
                description="Upload files from your computer"
                isSelected={selectedSource === 'files'}
                onClick={() => setSelectedSource('files')}
              />
            </div>

            {selectedSource === 's3' ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    S3 Bucket Name
                  </label>
                  <input
                    type="text"
                    value={sourceS3Bucket}
                    onChange={(e) => setSourceS3Bucket(e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="my-bucket"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    S3 Prefix (Optional)
                  </label>
                  <input
                    type="text"
                    value={sourceS3Prefix}
                    onChange={(e) => setSourceS3Prefix(e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="folder/path/"
                  />
                </div>
              </div>
            ) : (
              <div>
                <input
                  type="file"
                  multiple
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                  accept=".pdf, .txt, .csv, .html, .json, .xml"
                />
                <label
                  htmlFor="file-upload"
                  className="flex items-center justify-center w-full p-4 border-2 border-dashed border-gray-300 rounded-md cursor-pointer hover:border-gray-400"
                >
                  <div className="space-y-1 text-center">
                    <svg
                      className="mx-auto h-12 w-12 text-gray-400"
                      stroke="currentColor"
                      fill="none"
                      viewBox="0 0 48 48"
                    >
                      <path
                        d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4h-12m-4-12v8m0 0v8a4 4 0 01-4 4h-8m-4-12h8m-8 0v-8m32 0v-8"
                        strokeWidth="2"
                        strokeLinecap="round"
                      />
                    </svg>
                    <div className="flex text-sm text-gray-600">
                      <span className="relative rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-blue-500">
                        Select files
                      </span>
                    </div>
                    <p className="text-xs text-gray-500">Drag and drop files or click to browse</p>
                  </div>
                </label>

                {selectedFiles.length > 0 && (
                  <div className="mt-4">
                    <h3 className="text-sm font-medium text-gray-700">Selected Files ({selectedFiles.length})</h3>
                    <div className="mt-2 max-h-40 overflow-y-auto border border-gray-200 rounded-md p-2">
                      <ul className="space-y-1">
                        {selectedFiles.map((file, i) => (
                          <li key={i} className="flex items-center justify-between py-1">
                            <span className="text-sm text-gray-500">
                              {file.name} ({(file.size / 1024).toFixed(1)} KB)
                            </span>
                            <button
                              type="button"
                              onClick={() => removeFile(i)}
                              className="text-red-500 hover:text-red-700 p-1"
                              aria-label="Remove file"
                            >
                              <svg
                                className="h-4 w-4"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M6 18L18 6M6 6l12 12"
                                />
                              </svg>
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Tags Section - Only show if a model is selected */}
          {selectedModelId && (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-medium">Tags</h2>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={selectAllTags}
                  className="text-sm flex items-center"
                  disabled={isTagsLoading || selectedTags.length === availableTags.length}
                >
                  <span className="mr-1">Select All</span>
                  <input
                    type="checkbox"
                    checked={selectedTags.length === availableTags.length}
                    onChange={selectAllTags}
                    className="rounded border-gray-300"
                  />
                </Button>
              </div>

              {isTagsLoading ? (
                <div className="flex justify-center py-4">
                  <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : availableTags.length === 0 ? (
                <div className="text-gray-500 py-2">No tags available for this model</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {availableTags.map(tag => (
                    <Tag
                      key={tag}
                      tag={tag}
                      selected={selectedTags.includes(tag)}
                      onClick={() => toggleTag(tag)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Custom Tags Section */}
          <div>
            <h2 className="text-lg font-medium mb-4">Custom Tags</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {customTags.map((customTag) => (
                <div key={customTag.name} className="border border-gray-200 rounded-md overflow-hidden">
                  <div className="p-4 border-b border-gray-200 flex justify-between items-center">
                    <Tag tag={customTag.name} custom={true} selected />
                    <div className="flex items-center space-x-2">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleEditCustomTag(customTag)}
                        className="text-blue-500"
                      >
                        <Edit className="h-4 w-4 mr-1" />
                        Edit
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveCustomTag(customTag.name)}
                        className="text-red-500"
                      >
                        Remove
                      </Button>
                    </div>
                  </div>
                  <div className="p-4">
                    <p className="text-sm font-mono">{customTag.pattern}</p>
                  </div>
                </div>
              ))}

              {/* Add Custom Tag Card */}
              <div
                className="border border-dashed border-gray-300 rounded-md flex items-center justify-center p-6 cursor-pointer hover:border-gray-400"
                onClick={() => setIsCustomTagDialogOpen(true)}
              >
                <div className="flex flex-col items-center">
                  <Plus className="h-8 w-8 text-gray-400 mb-2" />
                  <span className="text-gray-600">Define new tag</span>
                </div>
              </div>
            </div>

            {isCustomTagDialogOpen && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
                <div className="bg-white rounded-lg p-6 w-full max-w-md">
                  <h3 className="text-lg font-medium mb-4">Create Custom Tag</h3>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Tag Name
                      </label>
                      <input
                        type="text"
                        value={customTagName}
                        onChange={(e) => setCustomTagName(e.target.value.toUpperCase())}
                        className="w-full p-2 border border-gray-300 rounded"
                        placeholder="CUSTOM_TAG_NAME"
                      />
                    </div>

                    <div>
                      <div className="flex items-center space-x-4 mb-1">
                        <label className="flex items-center text-sm text-gray-700">
                          <input
                            type="radio"
                            value="string"
                            checked={patternType === 'string'}
                            onChange={() => setPatternType('string')}
                            className="mr-1"
                          />
                          <span className="block text-sm font-medium text-gray-700">
                            String
                          </span>
                        </label>
                        <label className="flex items-center text-sm text-gray-700">
                          <input
                            type="radio"
                            value="regex"
                            checked={patternType === 'regex'}
                            onChange={() => setPatternType('regex')}
                            className="mr-1"
                          />
                          <span className="block text-sm font-medium text-gray-700">
                            Regex
                          </span>
                        </label>
                      </div>

                      <input
                        type="text"
                        value={customTagPattern}
                        onChange={(e) => setCustomTagPattern(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded"
                        placeholder={
                          patternType === 'regex' ? '\\b[A-Z]{2}\\d{6}\\b' : 'John Doe'
                        }
                      />

                      {patternType === 'string' && (
                        <p className="text-xs text-gray-500 mt-1">
                          Example: <code>John Doe</code> for matching an exact name
                        </p>
                      )}

                      {patternType === 'regex' && (
                        <p className="text-xs text-gray-500 mt-1">
                          Example: <code>\d{3}[-.]?\d{3}[-.]?\d{4}</code> for phone numbers
                        </p>
                      )}
                    </div>
                    <div className="flex justify-end space-x-2">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => setIsCustomTagDialogOpen(false)}
                      >
                        Cancel
                      </Button>
                      <Button type="button" onClick={handleAddCustomTag}>
                        Add Tag
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Groups Section */}
          <div>
            <h2 className="text-lg font-medium mb-4">Groups</h2>

            {/* Group creation form */}
            <div className="mb-4 p-4 border border-gray-200 rounded-md">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Group Name
                  </label>
                  <input
                    type="text"
                    value={groupName}
                    onChange={(e) => setGroupName(e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="sensitive_docs"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Group Query
                  </label>
                  <input
                    type="text"
                    value={groupQuery}
                    onChange={(e) => setGroupQuery(e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="COUNT(SSN) > 0"
                  />
                </div>
              </div>
              <Button
                onClick={handleAddGroup}
                disabled={!groupName || !groupQuery}
              >
                Add Group
              </Button>

              <div className="mt-4 text-sm text-gray-500">
                <p>Example queries:</p>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li><code>COUNT(SSN) &gt; 0</code> - Documents containing SSNs</li>
                  <li><code>COUNT(NAME) &gt; 2 AND COUNT(PHONE) &gt; 0</code> - Documents with multiple names and a phone number</li>
                </ul>
              </div>
            </div>

            {/* Display defined groups */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(groups).map(([name, query]) => (
                <GroupCard
                  key={name}
                  name={name}
                  definition={query}
                  onRemove={() => handleRemoveGroup(name)}
                />
              ))}
            </div>
          </div>

          {/* Submit Button */}
          <div className="flex justify-end pt-4">
            <Button
              type="submit"
              disabled={isSubmitting}
              className="w-full md:w-auto"
            >
              {isSubmitting ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : (
                'Create Job'
              )}
            </Button>
          </div>
        </form>
      )}
    </div>
  );
} 