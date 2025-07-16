'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Box } from '@mui/material';
import { ArrowLeft, Plus, RefreshCw, Edit, FileSearch } from 'lucide-react';
import { nerService } from '@/lib/backend';
import { NO_GROUP, uniqueFileNames, getFilesFromElectron } from '@/lib/utils';
import { Input } from '@/components/ui/input';
import { SearchIcon } from '@heroicons/react/solid';
import { useConditionalTelemetry } from '@/hooks/useConditionalTelemetry';
import { useEnterprise } from '@/hooks/useEnterprise';

import { nerBaseUrl } from '@/lib/axios.config';

const SUPPORTED_TYPES = ['.pdf', '.txt', '.csv', '.html', '.json', '.xml'];

// Tag chip component - reused from the detail page but with interactive mode
interface TagProps {
  tag: string;
  selected?: boolean;
  onClick?: () => void;
  custom?: boolean;
  addNew?: boolean;
}

const Tag: React.FC<TagProps> = ({
  tag,
  selected = false,
  onClick,
  custom = false,
  addNew = false,
}) => {
  return (
    <div
      className={`px-3 py-1 text-sm font-medium overflow-hidden text-ellipsis whitespace-nowrap max-w-[16vw] rounded-sm ${!custom && 'cursor-pointer'} ${selected ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
      style={{ userSelect: 'none' }}
      onClick={onClick}
      title={tag}
    >
      {tag}
    </div>
  );
};

// Source option card component - reused from the detail page
interface ModelOptionProps {
  title: string;
  description: React.ReactNode;
  isSelected?: boolean;
  disabled?: boolean;
  onClick: () => void;
}

const ModelOption: React.FC<ModelOptionProps> = ({
  title,
  description,
  isSelected = false,
  disabled = false,
  onClick,
}) => (
  <div
    className={`relative p-6 border rounded-md transition-all
      ${isSelected ? 'border-blue-500 border-2' : 'border-gray-200 border-2'}
      ${
        disabled
          ? 'opacity-85 cursor-not-allowed bg-gray-50'
          : 'cursor-pointer hover:border-blue-300'
      }
    `}
    onClick={() => !disabled && onClick()}
  >
    <h3 className="text-base font-medium">{title}</h3>
    <div className="text-sm text-gray-500 mt-1">{description}</div>
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

interface SourceOptionProps {
  onClick: () => void;
  input?: React.ReactNode;
  icon: React.ReactNode;
  title: string;
  description: string;
  disclaimer: string;
  disabled?: boolean;
}

const SourceOption: React.FC<SourceOptionProps> = ({
  onClick,
  input,
  icon,
  title,
  description,
  disclaimer,
  disabled = false,
}) => (
  <div
    className={`relative p-6 border-2 border-dashed rounded-lg transition-colors ${
      disabled
        ? 'border-gray-200 bg-gray-50 cursor-not-allowed opacity-60'
        : 'border-gray-300 hover:border-blue-400 cursor-pointer'
    }`}
    onClick={disabled ? () => {} : onClick}
  >
    {input && input}
    <div className="flex flex-col items-center justify-center space-y-4">
      <div className="flex items-center justify-center w-16 h-16 rounded-full bg-blue-50">
        <svg
          className="w-8 h-8 text-blue-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          {icon}
        </svg>
      </div>
      <div className="text-center">
        <h3 className="text-base font-medium mb-1">{title}</h3>
        <p className="text-sm text-gray-500">{description}</p>
        <p className="text-xs text-gray-400 mt-2">{disclaimer}</p>
      </div>
    </div>
  </div>
);

interface FileSourcesProps {
  selectSource: (source: 's3' | 'files' | 'directory') => void;
  isLoadingFiles: boolean;
  setIsLoadingFiles: (loading: boolean) => void;
  addFilesMeta: (filesMeta: any[]) => void;
}

const FileSources: React.FC<FileSourcesProps> = ({
  selectSource,
  isLoadingFiles,
  setIsLoadingFiles,
  addFilesMeta,
}) => {
  const s3 = (
    <SourceOption
      onClick={() => selectSource('s3')}
      icon={
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z"
        />
      }
      title="S3 Bucket"
      description="Scan files from an S3 bucket"
      disclaimer="Public buckets only without enterprise subscription."
      disabled={isLoadingFiles}
    />
  );

  const folderIcon = (
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
    />
  );

  // @ts-ignore
  if (window && window.electron) {
    // Check if we're on macOS
    const isMacOS = navigator.platform.toLowerCase().includes('mac');

    if (isMacOS) {
      // macOS: Single button that allows both files and folders
      return (
        <>
          <SourceOption
            onClick={async () => {
              selectSource('files');
              setIsLoadingFiles(true);
              try {
                const { allFilesMeta, totalSize, error } = await getFilesFromElectron(
                  SUPPORTED_TYPES,
                  false,
                  true
                ); // combined mode
                if (error) {
                  addFilesMeta([]);
                } else {
                  addFilesMeta(allFilesMeta || []);
                }
              } finally {
                setIsLoadingFiles(false);
              }
            }}
            icon={
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                className="lucide lucide-folder-icon lucide-folder"
              >
                <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
              </svg>
            }
            title="Local Files or Folders"
            description="Select files or folders from your computer"
            disclaimer={`Supported: ${SUPPORTED_TYPES.join(', ')}`}
            disabled={isLoadingFiles}
          />
          {s3}
        </>
      );
    } else {
      // Windows/Linux: Separate buttons for files and folders
      return (
        <>
          <SourceOption
            onClick={async () => {
              selectSource('files');
              setIsLoadingFiles(true);
              try {
                const { allFilesMeta, totalSize, error } = await getFilesFromElectron(
                  SUPPORTED_TYPES,
                  false
                );
                if (error) {
                  addFilesMeta([]);
                } else {
                  addFilesMeta(allFilesMeta || []);
                }
              } finally {
                setIsLoadingFiles(false);
              }
            }}
            icon={
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            }
            title="Local Files"
            description="Select individual files from your computer"
            disclaimer={`Supported: ${SUPPORTED_TYPES.join(', ')}`}
            disabled={isLoadingFiles}
          />
          <SourceOption
            onClick={async () => {
              selectSource('directory');
              setIsLoadingFiles(true);
              try {
                const { allFilesMeta, totalSize, error } = await getFilesFromElectron(
                  SUPPORTED_TYPES,
                  true
                );
                if (error) {
                  addFilesMeta([]);
                } else {
                  addFilesMeta(allFilesMeta || []);
                }
              } finally {
                setIsLoadingFiles(false);
              }
            }}
            icon={folderIcon}
            title="Local Directory"
            description="Select an entire folder to scan"
            disclaimer={`Supported: ${SUPPORTED_TYPES.join(', ')}`}
            disabled={isLoadingFiles}
          />
          {s3}
        </>
      );
    }
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setIsLoadingFiles(true);
      try {
        // Add a small delay to show loading state for quick file selections
        await new Promise((resolve) => setTimeout(resolve, 100));
        addFilesMeta(
          Array.from(files).map((file) => ({
            name: file.name,
            size: file.size,
            fullPath: '',
            file,
          }))
        );
      } finally {
        setIsLoadingFiles(false);
        e.target.value = '';
      }
    }
  };

  const fileInput = (
    <input
      type="file"
      id="file-upload"
      multiple
      onChange={handleFileChange}
      className="hidden"
      accept={SUPPORTED_TYPES.join(',')}
    />
  );

  const directoryInput = (
    <input
      type="file"
      id="directory-upload"
      {...({ webkitdirectory: '', directory: '' } as any)}
      onChange={handleFileChange}
      className="hidden"
      accept={SUPPORTED_TYPES.join(',')}
    />
  );

  return (
    <>
      <SourceOption
        onClick={() => {
          document.getElementById('file-upload')?.click();
          selectSource('files');
        }}
        input={fileInput}
        icon={
          isLoadingFiles ? (
            <RefreshCw className="w-8 h-8 animate-spin" />
          ) : (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          )
        }
        title="Local Files"
        description={isLoadingFiles ? 'Loading files...' : 'Scan files from your computer'}
        disclaimer={`Supported: ${SUPPORTED_TYPES.join(', ')}`}
        disabled={isLoadingFiles}
      />
      <SourceOption
        onClick={() => {
          document.getElementById('directory-upload')?.click();
          selectSource('directory');
        }}
        input={directoryInput}
        icon={isLoadingFiles ? <RefreshCw className="w-8 h-8 animate-spin" /> : folderIcon}
        title="Local Directory"
        description={isLoadingFiles ? 'Loading files...' : 'Scan an entire directory'}
        disclaimer={`Supported: ${SUPPORTED_TYPES.join(', ')}`}
        disabled={isLoadingFiles}
      />
      {s3}
    </>
  );
};

interface CustomTag {
  name: string;
  pattern: string;
}

export default function NewJobPage() {
  const router = useRouter();
  const recordEvent = useConditionalTelemetry();

  const { isEnterprise } = useEnterprise();

  // Essential state
  const [selectedSource, setSelectedSource] = useState<'s3' | 'files' | 'directory' | ''>('files');
  const [sourceS3Endpoint, setSourceS3Endpoint] = useState('');
  const [sourceS3Region, setSourceS3Region] = useState('');
  const [sourceS3Bucket, setSourceS3Bucket] = useState('');
  const [sourceS3Prefix, setSourceS3Prefix] = useState('');
  // [File object, full path] pairs. Full path may be empty if electron is not available.
  const [selectedFilesMeta, setSelectedFilesMeta] = useState<any[]>([]);

  // Helper to assign unique names to selectedFilesMeta
  const assignUniqueNames = (filesMeta: any[]) => {
    const names = filesMeta.map((f) => f.name);
    const uniqueNames = uniqueFileNames(names);
    return filesMeta.map((f, i) => ({ ...f, uniqueName: uniqueNames[i] }));
  };

  // Add files, deduplicate by fullPath, and assign unique names
  const addFilesMeta = (newFilesMeta: any[]) => {
    const allFiles = [...selectedFilesMeta, ...newFilesMeta];
    const seen = new Set<string>();
    const deduped = allFiles.filter((f) => {
      if (!f.fullPath) return true;
      if (seen.has(f.fullPath)) return false;
      seen.add(f.fullPath);
      return true;
    });
    setSelectedFilesMeta(assignUniqueNames(deduped));
  };

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [existingReportName, setExistingReportName] = useState<string[]>([]);
  const [jobName, setJobName] = useState('');

  const [models, setModels] = useState<any[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [modelSearchQuery, setModelSearchQuery] = useState('');

  const filteredCustomModels = useMemo(() => {
    return models
      .filter((model) => !['basic', 'advanced'].includes(model.Name.toLowerCase()))
      .filter((model) => model.Name.toLowerCase().includes(modelSearchQuery.toLowerCase()));
  }, [models, modelSearchQuery]);

  const defaultModels = useMemo(() => {
    if (isEnterprise) {
      return [
        {
          Id: models.find((model) => model.Name === 'basic')?.Id || 'basic',
          Name: 'Default',
          Disabled: false,
          Description:
            'Fast and lightweight AI model. Allows users to perpetually customize fields with user feedback, includes advanced monitoring features.',
        },
      ];
    }
    return [
      {
        Id: models.find((model) => model.Name === 'basic')?.Id || 'basic',
        Name: 'Basic',
        Disabled: false,
        Description:
          'Fast and lightweight AI model, comes with the free version, does not allow customization of the fields with user feedback, gives basic usage statistics.',
      },
      {
        Id: 'advanced',
        Name: 'Advanced',
        Disabled: true,
        Description:
          'Our most advanced AI model, available on enterprise platform. Allows users to perpetually customize fields with user feedback, includes advanced monitoring features.',
      },
    ];
  }, [models, isEnterprise]);

  // Tags handling
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [isTagsLoading, setIsTagsLoading] = useState(false);

  // Groups handling
  const [groupName, setGroupName] = useState('');
  const [groupQuery, setGroupQuery] = useState('');
  const [groups, setGroups] = useState<Record<string, string>>({});
  const [isGroupDialogOpen, setIsGroupDialogOpen] = useState(false);
  const [groupDialogError, setGroupDialogError] = useState<string | null>(null);
  const [editingGroup, setEditingGroup] = useState<{
    name: string;
    query: string;
  } | null>(null);

  // Custom tags handling
  const [customTags, setCustomTags] = useState<CustomTag[]>([]);
  const [customTagName, setCustomTagName] = useState('');
  const [customTagPattern, setCustomTagPattern] = useState('');
  const [isCustomTagDialogOpen, setIsCustomTagDialogOpen] = useState(false);
  const [editingTag, setEditingTag] = useState<CustomTag | null>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);

  // Error/Success messages
  const [error, setError] = useState<string | null>(null);
  const [s3Error, setS3Error] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const [patternType, setPatternType] = useState('string');

  const [isConfirmDialogOpen, setIsConfirmDialogOpen] = useState(false);

  const isFileSupported = (filename: string) => {
    return SUPPORTED_TYPES.some((ext) => filename.toLowerCase().endsWith(ext));
  };

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const modelData = await nerService.listModels();
        const trainedModels = modelData.filter((model) => model.Status === 'TRAINED');
        setModels(trainedModels.reverse());
        const basicModel = trainedModels.find((model) => model.Name === 'basic');
        setSelectedModelId(basicModel ? basicModel.Id : null);
      } catch (err) {
        console.error('Error fetching models:', err);
        setError('Failed to load models. Please try again.');
      }
    };

    fetchModels();

    const fetchReportNames = async () => {
      const response = await nerService.listReports();
      setExistingReportName(response.map((report) => report.ReportName));
    };
    fetchReportNames();
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
        console.log('Tags from model:', modelTags);
        const filteredModelTags = modelTags.filter((tag) => tag !== 'O');
        setAvailableTags(filteredModelTags);
        setSelectedTags(filteredModelTags); // By default, select all tags
      } catch (error) {
        console.error('Error fetching tags:', error);
        setError('Failed to load tags from the selected model');
      } finally {
        setIsTagsLoading(false);
      }
    };

    fetchTags();
  }, [selectedModelId]);

  const toggleTag = (tag: string) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter((t) => t !== tag));
    } else {
      setSelectedTags([...selectedTags, tag]);
    }
  };

  const selectAllTags = () => {
    setSelectedTags([...availableTags]);
  };

  const handleGroupCancel = () => {
    setGroupName('');
    setGroupQuery('');
    setEditingGroup(null);
    setGroupDialogError(null);
    setIsGroupDialogOpen(false);
  };

  const handleAddGroup = async () => {
    setGroupDialogError(null);

    if (!groupName.trim() || !groupQuery.trim()) {
      setGroupDialogError('Group name and query are required.');
      return;
    }

    if (!editingGroup && groups[groupName.toUpperCase()]) {
      setGroupDialogError('Group name already exists.');
      return;
    }

    const formattedGroupName = groupName.trim().toUpperCase();
    const formattedGroupQuery = groupQuery.trim().toUpperCase();

    if (formattedGroupName === NO_GROUP.trim().toUpperCase()) {
      setGroupDialogError(`Group name cannot be "${NO_GROUP}"`);
      return;
    }

    const errorMessage = await nerService.validateGroupDefinition(formattedGroupQuery);
    console.log('Error message:', errorMessage);

    if (errorMessage) {
      setGroupDialogError(errorMessage);
      return;
    }

    const nameExists = Object.keys(groups).some(
      (name) => name.toUpperCase() === formattedGroupName
    );

    if (nameExists) {
      setGroupDialogError(`Group name "${formattedGroupName}" already exists.`);
      return;
    }

    setGroups((prev) => {
      const updatedGroups = { ...prev };
      if (editingGroup && editingGroup.name !== formattedGroupName) {
        delete updatedGroups[editingGroup.name];
      }
      updatedGroups[formattedGroupName] = formattedGroupQuery;
      return updatedGroups;
    });

    setGroups((prev) => ({
      ...prev,
      [groupName.trim().toUpperCase()]: groupQuery.trim().toUpperCase(),
    }));

    handleGroupCancel();
  };

  const handleEditGroup = (name: string, query: string) => {
    setGroupName(name);
    setGroupQuery(query);
    setEditingGroup({ name, query });
    setIsGroupDialogOpen(true);
  };

  const handleRemoveGroup = (name: string) => {
    const newGroups = { ...groups };
    delete newGroups[name];
    setGroups(newGroups);
  };

  const handleAddCustomTag = () => {
    setDialogError(null);

    if (!customTagName.trim() || !customTagPattern.trim()) {
      setDialogError('Custom tag name and pattern are required');
      return;
    }

    const newCustomTag = {
      name: customTagName.trim().toUpperCase(),
      pattern: customTagPattern,
    };

    for (let index = 0; index < customTags.length; index++) {
      const thisTag = customTags[index];
      if (thisTag.name === customTagName.toUpperCase()) {
        setDialogError(`Custom Tag name "${customTagName}" already exists.`);
        return;
      }
    }

    if (editingTag) {
      setCustomTags((prev) =>
        prev.map((tag) => (tag.name === editingTag.name ? newCustomTag : tag))
      );
    } else {
      for (let index = 0; index < customTags.length; index++) {
        const thisTag = customTags[index];
        if (thisTag.name === customTagName.toUpperCase()) {
          setDialogError('Custom Tag name already exists.');
          return;
        }
      }
      setCustomTags((prev) => [...prev, newCustomTag]);
    }

    setCustomTagName('');
    setCustomTagPattern('');
    setEditingTag(null);
    setIsCustomTagDialogOpen(false);
  };
  const handleRemoveCustomTag = (tagName: string) => {
    setCustomTags((prev) => prev.filter((tag) => tag.name !== tagName));
    setAvailableTags((prev) => prev.filter((tag) => tag !== tagName));
    setSelectedTags((prev) => prev.filter((tag) => tag !== tagName));
  };

  const handleEditCustomTag = (tag: CustomTag) => {
    setCustomTagName(tag.name);
    setCustomTagPattern(tag.pattern);
    setEditingTag(tag);
    setIsCustomTagDialogOpen(true);
  };
  const handleCancel = () => {
    setCustomTagName('');
    setCustomTagPattern('');
    setEditingTag(null);
    setDialogError(null);
    setIsCustomTagDialogOpen(false);
  };

  const handleSelectFiles = async () => {
    setIsLoadingFiles(true);
    setError(null);
    try {
      const result = await getFilesFromElectron(SUPPORTED_TYPES);
      if (result.error) {
        setError(result.error);
        setSelectedFilesMeta([]);
      } else {
        addFilesMeta(result.allFilesMeta);
      }
    } catch (err: any) {
      setError('Failed to select files.');
      setSelectedFilesMeta([]);
    } finally {
      setIsLoadingFiles(false);
    }
  };

  // Handler for uploading files (calls main process IPC)

  const validateS3Bucket = async () => {
    if (!sourceS3Bucket || !sourceS3Region) {
      return;
    }
    setS3Error('');
    const s3Error = await nerService.attemptS3Connection(
      sourceS3Endpoint,
      sourceS3Region,
      sourceS3Bucket,
      sourceS3Prefix
    );
    if (s3Error) {
      setS3Error(s3Error);
    } else {
      setS3Error('');
    }
  };

  const removeFile = (index: number) => {
    setSelectedFilesMeta((prev) => prev.filter((_, i) => i !== index));
  };

  const handleCloseDialog = () => {
    setIsConfirmDialogOpen(false);
    // Reset the input value
    const input = document.getElementById(
      selectedSource === 'directory' ? 'directory-upload' : 'file-upload'
    ) as HTMLInputElement;
    if (input) {
      input.value = '';
    }
  };

  const validateCustomTagName = (name: string): boolean => {
    if (!name) {
      setDialogError('Tag name is required');
      return false;
    }

    if (!/^[A-Za-z0-9_]+$/.test(name)) {
      setDialogError('Tag name can only contain letters, numbers, and underscores');
      return false;
    }

    setDialogError(null);
    return true;
  };

  const handleTagNameChange = (name: string) => {
    const value = name.replace(/\s/g, '_');
    setCustomTagName(value);
    validateCustomTagName(value);
  };

  async function uploadFilesBrowser(filesMeta: any[], uploadUrl: string) {
    const formData = new FormData();
    filesMeta.forEach((meta) => {
      formData.append('files', meta.file, meta.uniqueName || meta.name);
    });
    const response = await fetch(uploadUrl, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      return { success: false, error: 'Upload failed' };
    }
    return await response.json();
  }

  // Submit the new job
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedModelId) {
      setError('Please select a model');
      return;
    }

    if (selectedSource === 's3' && !(sourceS3Region && sourceS3Bucket)) {
      setError('S3 region and bucket are required');
      return;
    }

    if (selectedSource === 's3' && s3Error) {
      setError(s3Error);
      return;
    }

    if (
      (selectedSource === 'files' || selectedSource === 'directory') &&
      selectedFilesMeta.length === 0
    ) {
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
      setIsSubmitting(true);
      setError(null);
      setSuccess(false);

      const uploadUrl = `${nerBaseUrl}/uploads`;
      let uploadId: string | undefined = undefined;

      // 1. Upload files via Electron main process or browser
      if (selectedSource === 'files' || selectedSource === 'directory') {
        // @ts-ignore
        if (typeof window !== 'undefined' && !window.electron) {
          // Browser context: upload using File objects
          const result = await uploadFilesBrowser(selectedFilesMeta, uploadUrl);

          if (!result.Id) {
            setError(result.error || 'Upload failed');
            setIsSubmitting(false);
            return;
          }
          uploadId = result.Id;

          // Note: file path mapping is not possible in browsers due to security reasons.
          if (typeof uploadId === 'string') {
            await nerService.storeFileNameToPath(uploadId, {});
          } else {
            throw new Error('uploadId is undefined when storing file name to path mapping');
          }
        } else {
          // Electron context: use main process
          // @ts-ignore
          const result = await window.electron.uploadFiles({
            filePaths: selectedFilesMeta.map((f) => f.fullPath),
            uploadUrl: uploadUrl,
            uniqueNames: selectedFilesMeta.map((f) => f.uniqueName),
            originalNames: selectedFilesMeta.map((f) => f.name),
          });

          if (!result.success || !result.uploadId) {
            setError(result.error || 'Upload failed');
            setIsSubmitting(false);
            return;
          }
          uploadId = result.uploadId;

          // 2. Store file path mappings (use uniqueName as key)
          const mapping: { [filename: string]: string } = {};
          selectedFilesMeta.forEach((fileMeta) => {
            if (fileMeta.fullPath) {
              mapping[fileMeta.uniqueName] = fileMeta.fullPath;
            }
          });
          console.log('Frontend - File path mapping being sent:', mapping);
          console.log('Frontend - Selected files meta:', selectedFilesMeta);
          console.log('Frontend - Upload ID:', uploadId);

          if (Object.keys(mapping).length > 0) {
            if (typeof uploadId === 'string') {
              await nerService.storeFileNameToPath(uploadId, mapping);
              console.log('Frontend - File path mapping sent successfully');
            } else {
              throw new Error('uploadId is undefined when storing file name to path mapping');
            }
          }
        }
      }
      const customTagsObj: Record<string, string> = {};
      customTags.forEach((tag) => {
        customTagsObj[tag.name] = tag.pattern;
      });

      // 3. Create the report, now that the files have been uploaded (or for S3, directly)
      const response = await nerService.createReport({
        ModelId: selectedModelId,
        Tags: selectedTags,
        CustomTags: customTagsObj,
        ...(selectedSource === 's3'
          ? {
              StorageType: 's3',
              StorageParams: {
                Endpoint: sourceS3Endpoint,
                Region: sourceS3Region,
                Bucket: sourceS3Bucket,
                Prefix: sourceS3Prefix,
              },
            }
          : {
              StorageType: 'upload',
              StorageParams: {
                UploadId: uploadId,
              },
            }),
        Groups: groups,
        ReportName: jobName,
      });

      setSuccess(true);

      recordEvent({
        UserAction: 'Create new report',
        UIComponent: 'Report Creation Form',
        Page: 'Report Creation Page',
      });

      // Redirect after success
      setTimeout(() => {
        router.push(`/token-classification/landing?tab=jobs`);
      }, 2000);
      setSelectedFilesMeta([]);
    } catch (err: unknown) {
      let errorMessage = 'An unexpected error occurred';

      if (
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as any).response?.data === 'string'
      ) {
        const data = (err as any).response.data;
        errorMessage = (data.charAt(0).toUpperCase() + data.slice(1)).trim();
      }

      setError(`Failed to create report. ${errorMessage}. Please try again.`);
    } finally {
      setIsSubmitting(false);
    }
  };
  const [isPressedSubmit, setIsPressedSubmit] = useState<boolean>(false);
  const [nameError, setNameError] = useState<string | null>(null);

  const validateJobName = (name: string): boolean => {
    if (!name) {
      setNameError('Scan name is required');
      return false;
    }

    if (existingReportName.includes(name)) {
      setNameError('Scan with this name already exists.');
      return false;
    }

    if (!/^[A-Za-z0-9_-]+$/.test(name)) {
      setNameError('Scan name can only contain letters, numbers, underscores, and hyphens');
      return false;
    }

    if (name.length > 50) {
      setNameError('Scan name must be less than 50 characters');
      return false;
    }

    setNameError(null);
    return true;
  };

  const [showTooltip, setShowTooltip] = useState<Record<string, boolean>>({});
  const copyToClipboard = (text: string, tooltipId: string) => {
    navigator.clipboard.writeText(text);
    setShowTooltip((prev) => ({ ...prev, [tooltipId]: true }));
    setTimeout(() => {
      setShowTooltip((prev) => ({ ...prev, [tooltipId]: false }));
    }, 1000);
  };

  const renderFileName = (file: File) => (
    <span className="text-sm text-gray-600">
      {file.name} ({(file.size / 1024).toFixed(1)} KB)
    </span>
  );

  const renderFileList = (files: File[]) => (
    <ul className="space-y-1">
      {files.map((file, i) => (
        <li key={i} className="flex items-center justify-between py-1">
          {renderFileName(file)}
          <button
            type="button"
            onClick={() => removeFile(i)}
            className="text-red-500 hover:text-red-700 p-1"
            aria-label="Remove file"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
  );

  return (
    <div className="container px-4 py-8" style={{ width: '90%' }}>
      {/* Title and Back Button */}
      <div className="flex items-center justify-between mb-6">
        <Button variant="outline" size="sm" asChild>
          <Link href={`/token-classification/landing?tab=jobs`} className="flex items-center">
            <ArrowLeft className="mr-1 h-4 w-4" /> Back to Scans
          </Link>
        </Button>
      </div>

      {error && !isPressedSubmit && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
          {error}
        </div>
      )}

      {success ? (
        <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded mb-6">
          Scanning in progress! Redirecting to Scans Dashboard...
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Job Name Field */}
          <Box className="bg-muted/60" sx={{ p: 3, borderRadius: 3 }}>
            <h2 className="text-2xl font-medium mb-4">Scan Name</h2>
            <div className="w-full">
              <input
                type="text"
                value={jobName}
                onChange={(e) => {
                  const value = e.target.value.replace(/\s/g, '_');
                  setJobName(value);
                  validateJobName(value);
                }}
                onBlur={() => validateJobName(jobName)}
                className={`w-full p-2 border ${
                  nameError ? 'border-red-500' : 'border-gray-300'
                } rounded`}
                placeholder="Enter_Scan_Name"
                required
              />
              {nameError ? (
                <p className="text-red-700 text-sm mt-1">
                  <sup className="text-red-700">*</sup>
                  {nameError}
                </p>
              ) : (
                <p className="text-sm text-gray-500 mt-1">
                  Use only letters, numbers, underscores, and hyphens. No spaces allowed.
                </p>
              )}
            </div>
          </Box>

          <Box sx={{ bgcolor: 'grey.100', p: 3, borderRadius: 3 }}>
            <h2 className="text-2xl font-medium mb-4">Source</h2>
            <div className="relative">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <FileSources
                  selectSource={setSelectedSource}
                  isLoadingFiles={isLoadingFiles}
                  setIsLoadingFiles={setIsLoadingFiles}
                  addFilesMeta={addFilesMeta}
                />
              </div>

              {/* Loading Overlay */}
              {isLoadingFiles && (
                <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded-lg z-10">
                  <div className="flex flex-col items-center space-y-3">
                    <RefreshCw className="h-8 w-8 animate-spin text-blue-500" />
                    <p className="text-sm font-medium text-gray-700">Loading files...</p>
                  </div>
                </div>
              )}
            </div>

            {selectedFilesMeta.length > 0 && (
              <div className="mt-6">
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-sm font-medium text-gray-700">
                    Selected Files ({selectedFilesMeta.length})
                  </h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedFilesMeta([])}
                    className="text-red-500"
                  >
                    Clear all
                  </Button>
                </div>
                <div className="max-h-60 overflow-y-auto border border-gray-200 rounded-md">
                  {selectedFilesMeta.map((fileMeta, index) => (
                    <div
                      key={fileMeta.fullPath || index}
                      className="flex items-center justify-between px-4 py-2 border-b last:border-b-0 hover:bg-gray-50"
                    >
                      <div className="flex items-center">
                        <div className="text-sm text-gray-600">
                          {fileMeta.fullPath || fileMeta.name}
                          <span className="text-xs text-gray-400 ml-2">
                            ({(fileMeta.size / 1024).toFixed(1)} KB)
                          </span>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => removeFile(index)}
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
                    </div>
                  ))}
                </div>
              </div>
            )}

            {selectedSource === 's3' && (
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-gray-500 mt-1 mb-3">
                    Your S3 bucket must be public. Private bucket support is available on the
                    enterprise platform. Reach out to{' '}
                    <div className="relative inline-block">
                      <span
                        className="text-blue-500 underline cursor-pointer hover:text-blue-700"
                        onClick={() => copyToClipboard('contact@thirdai.com', 's3')}
                        title="Click to copy email"
                      >
                        contact@thirdai.com
                      </span>
                      {showTooltip['s3'] && (
                        <div className="absolute left-1/2 -translate-x-1/2 mt-1 w-max px-2 py-1 text-xs bg-gray-800 text-white rounded shadow-md z-10">
                          Email Copied
                        </div>
                      )}
                    </div>{' '}
                    for an enterprise subscription.
                  </div>
                  {s3Error && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded">
                      {s3Error}
                    </div>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    S3 Endpoint (Optional)
                  </label>
                  <input
                    type="text"
                    value={sourceS3Endpoint}
                    onChange={(e) => setSourceS3Endpoint(e.target.value)}
                    onBlur={validateS3Bucket}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="s3.amazonaws.com"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">S3 Region</label>
                  <input
                    type="text"
                    value={sourceS3Region}
                    onChange={(e) => setSourceS3Region(e.target.value)}
                    onBlur={validateS3Bucket}
                    className="w-full p-2 border border-gray-300 rounded"
                    placeholder="us-east-1"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    S3 Bucket Name
                  </label>
                  <input
                    type="text"
                    value={sourceS3Bucket}
                    onChange={(e) => setSourceS3Bucket(e.target.value)}
                    onBlur={validateS3Bucket}
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
            )}
          </Box>

          {/* Model Selection */}
          <Box className="bg-muted/60" sx={{ p: 3, borderRadius: 3 }}>
            <h2 className="text-2xl font-medium mb-4">Model</h2>
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-4">Built-in Models</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {defaultModels.map((model) => (
                    <ModelOption
                      key={model.Id}
                      title={model.Name[0].toUpperCase() + model.Name.slice(1)}
                      description={model.Description || ''}
                      isSelected={selectedModelId === model.Id}
                      onClick={() => {
                        setSelectedModelId(model.Id);
                        setSelectedModel(model);
                      }}
                      disabled={model.Disabled}
                    />
                  ))}
                </div>
              </div>

              {models.filter((model) => !['basic', 'advanced'].includes(model.Name.toLowerCase()))
                .length > 0 && (
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold">Custom Models</h3>
                    <div className="relative w-64">
                      <input
                        type="text"
                        placeholder="Search custom models..."
                        value={modelSearchQuery}
                        onChange={(e) => setModelSearchQuery(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                      />
                      <SearchIcon className="absolute right-3 top-2.5 h-5 w-5 text-gray-400" />
                    </div>
                  </div>

                  {filteredCustomModels.length === 0 ? (
                    <div className="text-gray-500 py-8 text-center">
                      <FileSearch className="h-12 w-12 text-gray-400 mx-auto mb-3" />
                      <p className="text-sm">
                        No custom models found matching "{modelSearchQuery}"
                      </p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {filteredCustomModels
                        .filter((model) => model.Status !== 'TRAINING' && model.Status !== 'QUEUED')
                        .map((model) => (
                          <ModelOption
                            key={model.Id}
                            title={model.Name}
                            description={
                              <div className="space-y-2">
                                {model.BaseModelId && (
                                  <p className="text-sm text-gray-600">
                                    Base Model:{' '}
                                    {(() => {
                                      const baseModel = models.find(
                                        (m) => m.Id === model.BaseModelId
                                      );
                                      if (baseModel?.Name) {
                                        return (
                                          baseModel.Name.charAt(0).toUpperCase() +
                                          baseModel.Name.slice(1)
                                        );
                                      }
                                      return 'Unknown';
                                    })()}
                                  </p>
                                )}
                              </div>
                            }
                            isSelected={selectedModelId === model.Id}
                            onClick={() => {
                              setSelectedModelId(model.Id);
                              setSelectedModel(model);
                            }}
                          />
                        ))}
                    </div>
                  )}
                </div>
              )}

              {/* Tags Section - Only show if a model is selected */}
              {selectedModelId && (
                <div className="mt-8">
                  <div className="border-t pt-6">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold">Model Tags</h3>
                      <div className="flex space-x-2">
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
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedTags([])}
                          className="text-sm flex items-center"
                          disabled={isTagsLoading || selectedTags.length === 0}
                        >
                          <span className="mr-1">Clear Selection</span>
                          <input
                            type="checkbox"
                            checked={selectedTags.length === 0}
                            onChange={() => setSelectedTags([])}
                            className="rounded border-gray-300"
                          />
                        </Button>
                      </div>
                    </div>

                    {/* Added descriptive note */}
                    <p className="text-sm text-gray-500 mb-4">
                      Click on any tag to select/unselect it. By default, all tags are selected.
                    </p>

                    {isTagsLoading ? (
                      <div className="flex justify-center py-4">
                        <RefreshCw className="h-6 w-6 animate-spin text-gray-400" />
                      </div>
                    ) : availableTags.length === 0 ? (
                      <div className="text-gray-500 py-2">No tags available for this model</div>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {availableTags.map((tag) => (
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
                </div>
              )}
            </div>
          </Box>

          {/* Custom Tags Section */}
          <Box className="bg-muted/60" sx={{ p: 3, borderRadius: 3 }}>
            <h2 className="text-2xl font-medium mb-4">
              Custom Tags
              <span className="text-sm font-normal text-gray-500 ml-2">(Optional)</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {customTags.map((customTag) => (
                <div
                  key={customTag.name}
                  className="border border-gray-200 rounded-md overflow-hidden"
                >
                  <div className="py-1 px-2 border-b border-gray-200 flex justify-between items-center">
                    <Tag tag={customTag.name} custom={true} selected />
                    <div className="flex items-center space-x-3 px-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleEditCustomTag(customTag)}
                        className="text-blue-500 px-0"
                      >
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveCustomTag(customTag.name)}
                        className="text-red-500 px-0"
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
                className="border border-dashed border-gray-300 rounded-md flex items-center justify-center px-6 py-3 cursor-pointer hover:border-gray-400"
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
                  <h3 className="text-lg font-medium mb-4">
                    {`${editingTag ? 'Edit' : 'Create'} Custom Tag`}
                  </h3>
                  {dialogError && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded">
                      {dialogError}
                    </div>
                  )}
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Tag Name
                      </label>

                      <Input
                        id="tagName"
                        value={customTagName}
                        onChange={(e) => handleTagNameChange(e.target.value)}
                        onBlur={(e) => handleTagNameChange(e.target.value)}
                        className={`w-full p-2 border ${
                          nameError ? 'border-red-500' : 'border-gray-300'
                        } rounded`}
                        placeholder="CUSTOM_TAG_NAME"
                        required
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
                          <span className="block text-sm font-medium text-gray-700">String</span>
                        </label>
                        <label className="flex items-center text-sm text-gray-700">
                          <input
                            type="radio"
                            value="regex"
                            checked={patternType === 'regex'}
                            onChange={() => setPatternType('regex')}
                            className="mr-1"
                          />
                          <span className="block text-sm font-medium text-gray-700">Regex</span>
                        </label>
                      </div>

                      <input
                        type="text"
                        value={customTagPattern}
                        onChange={(e) => setCustomTagPattern(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded"
                        placeholder={patternType === 'regex' ? '\\b[A-Z]{2}\\d{6}\\b' : 'John Doe'}
                      />

                      {patternType === 'string' && (
                        <div className="text-sm text-gray-500">
                          <p>Example queries:</p>
                          <ul className="list-disc pl-5 mt-1 space-y-1">
                            <li>
                              <code>John Doe</code> for an exact string
                            </li>
                            <li>
                              <code>Alice|Bob</code> for a list of strings
                            </li>
                          </ul>
                        </div>
                      )}

                      {patternType === 'regex' && (
                        <p className="text-xs text-gray-500 mt-1">
                          Example:{' '}
                          <code>
                            \d{3}[-.]?\d{3}[-.]?\d{4}
                          </code>{' '}
                          for phone numbers
                        </p>
                      )}
                    </div>
                    <div className="flex justify-end space-x-2">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={handleCancel}
                        style={{
                          color: '#1976d2',
                        }}
                      >
                        Cancel
                      </Button>
                      <Button
                        type="button"
                        variant="default"
                        color="primary"
                        style={{
                          backgroundColor: '#1976d2',
                          textTransform: 'none',
                          fontWeight: 500,
                        }}
                        onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#1565c0')}
                        onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#1976d2')}
                        onClick={handleAddCustomTag}
                      >
                        Add Tag
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </Box>

          {/* Groups Section */}
          <Box className="bg-muted/60" sx={{ p: 3, borderRadius: 3 }}>
            <h2 className="text-2xl font-medium mb-4">
              Groups
              <span className="text-sm font-normal text-gray-500 ml-2">(Optional)</span>
            </h2>

            {/* Display defined groups */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {Object.entries(groups).map(([name, query]) => (
                <div key={name} className="border border-gray-200 rounded-md overflow-hidden">
                  <div className="py-1 px-4 border-b border-gray-200 flex justify-between items-center">
                    <span className="font-medium">{name}</span>
                    <div className="flex items-center space-x-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleEditGroup(name, query)}
                        className="text-blue-500"
                      >
                        <Edit className="h-4 w-4 mr-1" />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveGroup(name)}
                        className="text-red-500"
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
                      </Button>
                    </div>
                  </div>
                  <div className="p-4">
                    <p className="text-sm font-mono">{query}</p>
                  </div>
                </div>
              ))}

              {/* Add New Group Card */}
              <div
                className="border border-dashed border-gray-300 rounded-md flex items-center justify-center px-6 py-3 cursor-pointer hover:border-gray-400"
                onClick={() => setIsGroupDialogOpen(true)}
              >
                <div className="flex flex-col items-center">
                  <Plus className="h-8 w-8 text-gray-400 mb-2" />
                  <span className="text-gray-600">Define new group</span>
                </div>
              </div>
            </div>

            {/* Group Dialog */}
            {isGroupDialogOpen && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
                <div className="bg-white rounded-lg p-6 w-full max-w-md">
                  <h3 className="text-lg font-medium mb-4">
                    {editingGroup ? 'Edit Group' : 'Create Group'}
                  </h3>

                  {groupDialogError && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded">
                      {groupDialogError}
                    </div>
                  )}

                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Group Name
                      </label>
                      <Input
                        type="text"
                        value={groupName}
                        onChange={(e) => {
                          const value = e.target.value.replace(/\s/g, '_');
                          setGroupName(value);
                        }}
                        className="w-full p-2 border border-gray-300 rounded"
                        placeholder="sensitive_docs"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Group Query
                      </label>
                      <Input
                        type="text"
                        value={groupQuery}
                        onChange={(e) => setGroupQuery(e.target.value)}
                        className="w-full p-2 border border-gray-300 rounded"
                        placeholder="COUNT(SSN) > 0"
                      />
                    </div>

                    <div className="text-sm text-gray-500">
                      <p>Example queries:</p>
                      <ul className="list-disc pl-5 mt-1 space-y-1">
                        <li>
                          <code>COUNT(SSN) &gt; 0</code> - Documents containing SSNs
                        </li>
                        <li>
                          <code>COUNT(NAME) &gt; 2 AND COUNT(PHONE) &gt; 0</code> - Documents with
                          multiple names and a phone number
                        </li>
                      </ul>
                    </div>

                    <div className="flex justify-end space-x-2">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={handleGroupCancel}
                        style={{
                          color: '#1976d2',
                        }}
                      >
                        Cancel
                      </Button>
                      <Button
                        type="button"
                        variant="default"
                        color="primary"
                        style={{
                          backgroundColor: '#1976d2',
                          textTransform: 'none',
                          fontWeight: 500,
                        }}
                        onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#1565c0')}
                        onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#1976d2')}
                        onClick={handleAddGroup}
                      >
                        {editingGroup ? 'Save Changes' : 'Add Group'}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </Box>

          {/* Submit Button */}
          <div className="flex flex-col items-center space-y-4 pt-4">
            {error && isPressedSubmit && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded w-full max-w-md text-center">
                {error}
              </div>
            )}
            <Button
              variant="default"
              color="primary"
              style={{
                backgroundColor: '#1976d2',
                textTransform: 'none',
                fontWeight: 500,
              }}
              onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#1565c0')}
              onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#1976d2')}
              onClick={() => {
                setIsPressedSubmit(true);
              }}
            >
              {isSubmitting ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : (
                'Create Scan'
              )}{' '}
            </Button>
          </div>
        </form>
      )}

      {isConfirmDialogOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-medium mb-4">No Supported Files</h3>
            <p className="text-sm text-amber-600 mb-4">
              No supported files found in the selected directory.
              <br />
              Only {SUPPORTED_TYPES.join(', ')} files are supported.
            </p>
            <div className="flex justify-end">
              <Button
                type="button"
                variant="default"
                onClick={handleCloseDialog}
                style={{
                  backgroundColor: '#1976d2',
                  color: 'white',
                }}
              >
                OK
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
