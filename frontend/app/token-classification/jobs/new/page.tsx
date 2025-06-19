'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Box } from '@mui/material';
import { ArrowLeft, Plus, RefreshCw, Edit } from 'lucide-react';
import { nerService } from '@/lib/backend';
import { NO_GROUP, uniqueFileNames, getFilesFromElectron } from '@/lib/utils';
import { Input } from '@/components/ui/input';

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
      className={`px-3 py-1 text-sm font-medium overflow-x-scroll max-w-[16vw] rounded-sm ${!custom && 'cursor-pointer'} ${selected ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
      style={{ userSelect: 'none' }}
      onClick={onClick}
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
  handleLocalFiles: (files: [File, string][], isUploaded: boolean) => void;
  isLoadingFiles: boolean;
  setIsLoadingFiles: (loading: boolean) => void;
  setSelectedFilesMeta: React.Dispatch<React.SetStateAction<any[]>>;
}

const FileSources: React.FC<FileSourcesProps> = ({
  selectSource,
  handleLocalFiles,
  isLoadingFiles,
  setIsLoadingFiles,
  setSelectedFilesMeta,
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
    return (
      <>
        <SourceOption
          onClick={async () => {
            selectSource('files');
            setIsLoadingFiles(true);
            try {
              const { allFilesMeta, totalSize, error } =
                await getFilesFromElectron(SUPPORTED_TYPES);
              if (error) {
                // setError(error);
                setSelectedFilesMeta([]);
              } else {
                setSelectedFilesMeta(allFilesMeta || []);
                console.log('Total size of selected files:', totalSize / (1024 * 1024), 'MB');
              }
            } finally {
              setIsLoadingFiles(false);
            }
          }}
          icon={isLoadingFiles ? <RefreshCw className="w-8 h-8 animate-spin" /> : folderIcon}
          title="Local Files"
          description={isLoadingFiles ? 'Loading files...' : 'Scan files from your computer'}
          disclaimer={`Supported: ${SUPPORTED_TYPES.join(', ')}`}
          disabled={isLoadingFiles}
        />
        {s3}
      </>
    );
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      setIsLoadingFiles(true);
      try {
        // Add a small delay to show loading state for quick file selections
        await new Promise((resolve) => setTimeout(resolve, 100));
        handleLocalFiles(
          Array.from(files).map((file) => [file, '']),
          true
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

// Custom Tag interface
interface CustomTag {
  name: string;
  pattern: string;
}

export default function NewJobPage() {
  const router = useRouter();

  // Essential state
  const [selectedSource, setSelectedSource] = useState<'s3' | 'files' | 'directory' | ''>('files');
  const [sourceS3Endpoint, setSourceS3Endpoint] = useState('');
  const [sourceS3Region, setSourceS3Region] = useState('');
  const [sourceS3Bucket, setSourceS3Bucket] = useState('');
  const [sourceS3Prefix, setSourceS3Prefix] = useState('');
  // [File object, full path] pairs. Full path may be empty if electron is not available.
  const [selectedFilesMeta, setSelectedFilesMeta] = useState<any[]>([]);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [existingReportName, setExistingReportName] = useState<string[]>([]);
  //Job Name
  const [jobName, setJobName] = useState('');

  // Model selection
  const [models, setModels] = useState<any[]>([]);
  //Bi-default Presidio model is selected.
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<any>(null);

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

  // Fetch models on page load
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const modelData = await nerService.listModels();
        // Only show trained models that can be used for inference
        const trainedModels = modelData.filter((model) => model.Status === 'TRAINED');
        setModels(trainedModels.reverse());
        setSelectedModelId(trainedModels[0].Id);
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

  // Toggle tag selection
  const toggleTag = (tag: string) => {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter((t) => t !== tag));
    } else {
      setSelectedTags([...selectedTags, tag]);
    }
  };

  // Select all tags
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

    if (!editingGroup && groups[groupName]) {
      setGroupDialogError('Group name must be unique.');
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

  // Add a custom tag
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

  // const addFiles = (files: [File, string][]) => {
  //   const newSelectedFiles = [...selectedFiles];

  //   files.forEach(([newFile, newFullPath]) => {
  //     const existingIndex = newSelectedFiles.findIndex((existingFile) => {
  //       // In practice, either both are empty or both are not empty
  //       // If both are not empty, it means we are using electron to choose files
  //       // Otherwise, we are using the file input to choose files
  //       if (existingFile[1] !== '' && newFullPath !== '') {
  //         return existingFile[1] === newFullPath;
  //       }
  //       return existingFile[0].name === newFile.name;
  //     });

  //     if (existingIndex !== -1) {
  //       // Duplicate file so, replace the existing file with the new one
  //       newSelectedFiles[existingIndex] = [newFile, newFullPath];
  //     } else {
  //       // Add the new file
  //       newSelectedFiles.push([newFile, newFullPath]);
  //     }
  //   });

  //   // This is to handle the case where there are multiple files with the same name
  //   // but different full paths.
  //   const newFileNames = uniqueFileNames(newSelectedFiles.map((file) => file[0].name));

  //   setSelectedFiles(
  //     newSelectedFiles.map(([file, fullPath], index) => {
  //       const newFile = new File([file], newFileNames[index], {
  //         type: file.type,
  //         lastModified: file.lastModified,
  //       });
  //       return [newFile, fullPath];
  //     })
  //   );
  // };

  // // Update file handling to use file/directory input
  // const handleLocalFiles = (files: [File, string][], isUploaded: boolean) => {
  //   const supportedFiles = files.filter((file) => isFileSupported(file[0].name));

  //   if (supportedFiles.length > 0) {
  //     addFiles(supportedFiles);
  //   } else {
  //     if (isUploaded) setIsConfirmDialogOpen(true);
  //   }
  // };

  const getFilesFromElectron = async (supportedTypes: string[]) => {
    // @ts-ignore
    const results = await window.electron.openFileChooser(
      supportedTypes.map((t) => t.replace('.', ''))
    );
    if (results.error) {
      return { allFilesMeta: [], error: results.error };
    }
    return { allFilesMeta: results.allFilesMeta };
  };

  // Handler for selecting files/folders
  const handleSelectFiles = async () => {
    setIsLoadingFiles(true);
    setError(null);
    try {
      const result = await getFilesFromElectron(SUPPORTED_TYPES);
      if (result.error) {
        setError(result.error);
        setSelectedFilesMeta([]);
      } else {
        setSelectedFilesMeta(result.allFilesMeta);
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

      // 1. Upload files via Electron main process
      const filePaths = selectedFilesMeta.map((f) => f.fullPath);
      const uploadUrl = `${nerBaseUrl}/uploads`;

      // @ts-ignore
      const result = await window.electron.invoke('upload-files', { filePaths, uploadUrl });

      if (!result.success || !result.uploadId) {
        setError(result.error || 'Upload failed');
        setIsSubmitting(false);
        return;
      }

      console.log('Upload result:', result);

      // 2. Store file path mappings if needed
      const mapping: { [filename: string]: string } = {};
      selectedFilesMeta.forEach((fileMeta) => {
        if (fileMeta.fullPath) {
          mapping[fileMeta.name] = fileMeta.fullPath;
        }
      });
      if (Object.keys(mapping).length > 0) {
        await nerService.storeFileNameToPath(result.uploadId, mapping);
        console.log('stored upload paths', mapping);
      }
      const customTagsObj: Record<string, string> = {};
      customTags.forEach((tag) => {
        customTagsObj[tag.name] = tag.pattern;
      });

      // 3. Create the report
      const response = await nerService.createReport({
        ModelId: selectedModelId,
        Tags: selectedTags,
        CustomTags: customTagsObj,
        ...(selectedSource === 's3'
          ? {
              S3Endpoint: sourceS3Endpoint,
              S3Region: sourceS3Region,
              SourceS3Bucket: sourceS3Bucket,
              SourceS3Prefix: sourceS3Prefix || undefined,
            }
          : {
              UploadId: result.uploadId,
            }),
        Groups: groups,
        report_name: jobName,
      });

      setSuccess(true);
      setSelectedFilesMeta([]);
      // Optionally redirect or show a message
    } catch (err: any) {
      setError('Failed to create report. Please try again.');
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }

    // try {
    //   const filePaths = selectedFilesMeta.map((f) => f.fullPath);
    //   const uploadUrl = `${nerBaseUrl}/uploads`;

    //   // @ts-ignore
    //   const result = await window.electron.invoke('upload-files', { filePaths, uploadUrl });
    //   if (result.success) {
    //     setSuccess(true);
    //     setSelectedFilesMeta([]);
    //   } else {
    //     setError(result.error || 'Upload failed');
    //   }
    //   // let uploadId: string | undefined;

    //   // // Handle file/directory uploads if needed
    //   // if (selectedSource === 'files' || selectedSource === 'directory') {
    //   //   const uploadResponse = await nerService.uploadFiles(
    //   //     selectedFilesMeta.map((file) => file)
    //   //   );
    //   //   uploadId = uploadResponse.Id;

    //   //   // Store file path mappings for local uploads if full path is available
    //   //   const mapping: { [filename: string]: string } = {};
    //   //   selectedFilesMeta.forEach((fileMeta) => {
    //   //     if (fileMeta.fullPath) {
    //   //       mapping[fileMeta.name] = fileMeta.fullPath;
    //   //     }
    //   //   });
    //   //   if (Object.keys(mapping).length > 0) {
    //   //     await nerService.storeFileNameToPath(uploadId, mapping);
    //   //     console.log('stored upload paths', mapping);
    //   //   }
    //   // }

    //   // Create custom tags object for API
    //   const customTagsObj: Record<string, string> = {};
    //   customTags.forEach((tag) => {
    //     customTagsObj[tag.name] = tag.pattern;
    //   });

    //   // Create the report
    //   console.log('Job Name: ', jobName);
    //   const response = await nerService.createReport({
    //     ModelId: selectedModelId,
    //     Tags: selectedTags,
    //     CustomTags: customTagsObj,
    //     ...(selectedSource === 's3'
    //       ? {
    //           S3Endpoint: sourceS3Endpoint,
    //           S3Region: sourceS3Region,
    //           SourceS3Bucket: sourceS3Bucket,
    //           SourceS3Prefix: sourceS3Prefix || undefined,
    //         }
    //       : {
    //           UploadId: result.uploadId,
    //         }),
    //     Groups: groups,
    //     report_name: jobName,
    //   });

    //   setSuccess(true);

    //   // Redirect after success
    //   setTimeout(() => {
    //     router.push(`/token-classification/landing?tab=jobs`);
    //   }, 2000);
    // } catch (err: unknown) {
    //   let errorMessage = 'An unexpected error occurred';

    //   if (
    //     typeof err === 'object' &&
    //     err !== null &&
    //     'response' in err &&
    //     typeof (err as any).response?.data === 'string'
    //   ) {
    //     const data = (err as any).response.data;
    //     errorMessage = (data.charAt(0).toUpperCase() + data.slice(1)).trim();
    //   }

    //   setError(`Failed to create report. ${errorMessage}. Please try again.`);
    //   console.error(err);
    // } finally {
    //   setIsSubmitting(false);
    // }
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
                  handleLocalFiles={handleSelectFiles}
                  isLoadingFiles={isLoadingFiles}
                  setIsLoadingFiles={setIsLoadingFiles}
                  setSelectedFilesMeta={setSelectedFilesMeta}
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
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeFile(index)}
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
            <div>
              <h2 className="text-2xl font-medium mb-4">Model</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {models.map((model) => (
                  <ModelOption
                    key={model.Id}
                    title={model.Name[0].toUpperCase() + model.Name.slice(1)}
                    description={
                      'Fast and lightweight AI model, comes with the free version, does not allow customization of the fields with user feedback, gives basic usage statistics.'
                    }
                    isSelected={selectedModelId === model.Id}
                    onClick={() => setSelectedModelId(model.Id)}
                    disabled={model.Name === 'presidio'}
                  />
                ))}
                <ModelOption
                  key={'Advanced-Model'}
                  title={'Advanced'}
                  description={
                    <>
                      Our most advanced AI model, available on enterprise platform. Allows users to
                      perpetually customize fields with user feedback, includes advanced monitoring
                      features. Reach out to{' '}
                      <div className="relative inline-block">
                        <span
                          className="text-blue-500 underline cursor-pointer hover:text-blue-700"
                          onClick={() => copyToClipboard('contact@thirdai.com', 'advanced-model')}
                          title="Click to copy email"
                        >
                          contact@thirdai.com
                        </span>
                        {showTooltip['advanced-model'] && (
                          <div className="absolute left-1/2 -translate-x-1/2 mt-1 w-max px-2 py-1 text-xs bg-gray-800 text-white rounded shadow-md z-10">
                            Email Copied
                          </div>
                        )}
                      </div>{' '}
                      for an enterprise subscription.
                    </>
                  }
                  isSelected={false}
                  onClick={() => {}}
                  disabled={true}
                />
              </div>
            </div>

            {/* Tags Section - Only show if a model is selected */}
            {selectedModelId && (
              <div>
                <div className="flex justify-between items-center my-2">
                  <h2 className="text-lg font-medium">Tags</h2>
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
            )}
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
