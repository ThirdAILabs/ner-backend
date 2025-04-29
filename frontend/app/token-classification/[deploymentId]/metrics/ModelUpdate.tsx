'use client';

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Alert,
  Tooltip
} from '@mui/material';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Upload, HelpCircle } from 'lucide-react';
import RecentSamples from './RecentSamples';
// Comment out the import that's causing errors until the component is created
// import { TrainingResults } from './MetricsChart';
// import type { TrainReportData } from '@/lib/backend';

// Mock functions to simulate API calls
// These would be replaced with actual API calls in a real implementation
const getTrainReport = async (modelId: string) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  return {
    data: {
      metrics: {
        before: { precision: 0.85, recall: 0.82, f1: 0.83 },
        after: { precision: 0.91, recall: 0.89, f1: 0.90 }
      }
    }
  };
};

const getLabels = async () => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 600));
  
  return ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE', 'O', 'PHONE', 'EMAIL', 'ADDRESS', 'PRODUCT', 'EVENT', 'TIME', 'MONEY', 'PERCENT', 'QUANTITY'];
};

const trainUDTWithCSV = async ({ model_name, file, base_model_identifier, test_split }: any) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  return {
    status: 'success',
    message: 'Training initiated successfully',
    model_id: 'model_123456'
  };
};

const retrainTokenClassifier = async ({ model_name, base_model_id }: any) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  return {
    status: 'success',
    message: 'Retraining initiated successfully',
    model_id: 'model_789012'
  };
};

interface ModelUpdateProps {
  username?: string;
  modelName?: string;
  deploymentUrl?: string;
  workflowNames?: string[];
  deployStatus?: string;
  modelId?: string;
}

const ModelUpdate: React.FC<ModelUpdateProps> = ({
  username = 'user',
  modelName = 'token-classifier',
  deploymentUrl = 'https://api.example.com/token-classification',
  workflowNames = [],
  deployStatus = 'complete',
  modelId = 'model_123'
}) => {
  // States for CSV upload
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploadUpdating, setIsUploadUpdating] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [newModelName, setNewModelName] = useState(``);
  const [warningMessage, setWarningMessage] = useState('');

  // States for polling method
  const [isPollingUpdating, setIsPollingUpdating] = useState(false);
  const [pollingError, setPollingError] = useState('');
  const [pollingSuccess, setPollingSuccess] = useState(false);

  // States for training report
  const [trainReport, setTrainReport] = useState<any | null>(null);
  const [isLoadingReport, setIsLoadingReport] = useState(true);
  const [reportError, setReportError] = useState('');

  // New states for button cooldown
  const [uploadButtonDisabled, setUploadButtonDisabled] = useState(false);
  const [pollingButtonDisabled, setPollingButtonDisabled] = useState(false);

  // State for toggling tags list
  const [numTagDisplay, setNumTagDisplay] = useState<number>(5);
  const [tags, setTags] = useState<string[]>([]);

  // New state for polling model name
  const [pollingModelName, setPollingModelName] = useState(``);
  const [pollingWarningMessage, setPollingWarningMessage] = useState('');

  // Effect to validate model name on each change
  useEffect(() => {
    validateModelName(newModelName);
  }, [newModelName, workflowNames]);

  // Effect to validate polling model name
  useEffect(() => {
    validatePollingModelName(pollingModelName);
  }, [pollingModelName, workflowNames]);

  // Timer effect for upload button cooldown
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (uploadButtonDisabled) {
      timer = setTimeout(() => {
        setUploadButtonDisabled(false);
      }, 3000);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [uploadButtonDisabled]);

  // Timer effect for polling button cooldown
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (pollingButtonDisabled) {
      timer = setTimeout(() => {
        setPollingButtonDisabled(false);
      }, 3000);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [pollingButtonDisabled]);

  // Fetch initial training report
  useEffect(() => {
    const fetchInitialReport = async () => {
      try {
        setIsLoadingReport(true);
        setReportError('');
        const response = await getTrainReport(modelId);
        setTrainReport(response.data);
      } catch (error) {
        setReportError(error instanceof Error ? error.message : 'Failed to fetch training report');
      } finally {
        setIsLoadingReport(false);
      }
    };

    fetchInitialReport();
  }, [modelId]);

  // Fetch labels/tags
  useEffect(() => {
    const fetchTags = async () => {
      try {
        const labels = await getLabels();
        const filteredLabels = labels.filter((label) => label !== 'O');
        setTags(filteredLabels);
      } catch (error) {
        console.error('Error fetching labels:', error);
      }
    };
    
    fetchTags();
  }, []);

  const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB in bytes

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      // Check file type
      if (file.type !== 'text/csv') {
        setUploadError('Please upload a CSV file');
        setSelectedFile(null);
        return;
      }

      // Check file size
      if (file.size > MAX_FILE_SIZE) {
        setUploadError('File size must be less than 500MB');
        setSelectedFile(null);
        return;
      }

      setSelectedFile(file);
      setUploadError('');
    }
  };

  const validateModelName = (name: string) => {
    // Check if name exists in workflowNames
    if (workflowNames.includes(name)) {
      setWarningMessage(
        'An app with the same name already exists. Please choose a different name.'
      );
      return false;
    }

    // Check for valid characters (alphanumeric, hyphens, and underscores)
    const isValid = /^[a-zA-Z0-9-_]+$/.test(name);
    const isNotEmpty = name.trim().length > 0;

    if (!isValid && isNotEmpty) {
      setWarningMessage(
        'The app name can only contain letters, numbers, underscores, and hyphens. Please modify the name.'
      );
      return false;
    }

    if (name.includes(' ')) {
      setWarningMessage('The app name cannot contain spaces. Please remove the spaces.');
      return false;
    }

    if (name.includes('.')) {
      setWarningMessage("The app name cannot contain periods ('.'). Please remove the periods.");
      return false;
    }

    setWarningMessage('');
    return isValid && isNotEmpty;
  };

  const validatePollingModelName = (name: string) => {
    if (workflowNames.includes(name)) {
      setPollingWarningMessage(
        'An app with the same name already exists. Please choose a different name.'
      );
      return false;
    }
    const isValid = /^[a-zA-Z0-9-_]+$/.test(name);
    const isNotEmpty = name.trim().length > 0;
    if (!isValid && isNotEmpty) {
      setPollingWarningMessage(
        'The app name can only contain letters, numbers, underscores, and hyphens.'
      );
      return false;
    }
    if (name.includes(' ')) {
      setPollingWarningMessage('The app name cannot contain spaces.');
      return false;
    }
    if (name.includes('.')) {
      setPollingWarningMessage("The app name cannot contain periods ('.').");
      return false;
    }
    setPollingWarningMessage('');
    return isValid && isNotEmpty;
  };

  const handleUploadUpdate = async () => {
    if (!selectedFile) {
      setUploadError('Please select a CSV file first');
      return;
    }
    if (!validateModelName(newModelName)) {
      setUploadError(
        'Please enter a valid model name (alphanumeric characters, hyphens, and underscores only)'
      );
      return;
    }

    setIsUploadUpdating(true);
    setUploadError('');
    setUploadSuccess(false);
    setUploadButtonDisabled(true);

    try {
      const response = await trainUDTWithCSV({
        model_name: newModelName,
        file: selectedFile,
        base_model_identifier: `${username}/${modelName}`,
        test_split: 0.1,
      });

      if (response.status === 'success') {
        setUploadSuccess(true);
      } else {
        throw new Error(response.message || 'Failed to initiate update');
      }
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : 'An error occurred while initiating the model update'
      );
    } finally {
      setIsUploadUpdating(false);
    }
  };

  const handlePollingUpdate = async () => {
    if (!validatePollingModelName(pollingModelName)) {
      setPollingError('Please enter a valid model name for the new model.');
      return;
    }

    // Check deploy status before proceeding
    if (deployStatus !== 'complete') {
      setPollingError('Model must be fully deployed before updating with user feedback.');
      return;
    }

    setIsPollingUpdating(true);
    setPollingError('');
    setPollingSuccess(false);
    setPollingButtonDisabled(true);
    
    try {
      const response = await retrainTokenClassifier({
        model_name: pollingModelName,
        base_model_id: modelId,
      });
      
      if (response.status === 'success') {
        setPollingSuccess(true);
      } else {
        throw new Error(response.message || 'Failed to initiate update');
      }
    } catch (error) {
      setPollingError(
        error instanceof Error
          ? error.message
          : 'An error occurred while initiating the model update'
      );
    } finally {
      setIsPollingUpdating(false);
    }
  };

  const handleTagDisplayMore = () => {
    setNumTagDisplay(tags.length);
  };

  const handleTagDisplayLess = () => {
    setNumTagDisplay(5);
  };

  return (
    <div className="space-y-6 px-1">
      {/* Training Report Section */}
      {isLoadingReport ? (
        <Card>
          <CardContent>
            <div className="text-center py-8">Loading training report...</div>
          </CardContent>
        </Card>
      ) : reportError ? (
        <></>
      ) : (
        trainReport && (
          <Card>
            <CardContent>
              <div className="text-center py-4">
                <Typography variant="h6">Training Report</Typography>
                <div className="mt-2">
                  <div>Before: F1 {trainReport.metrics?.before?.f1.toFixed(2) || 'N/A'}</div>
                  <div>After: F1 {trainReport.metrics?.after?.f1.toFixed(2) || 'N/A'}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )
      )}

      {/* CSV Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle>Update Model with your own data</CardTitle>
          <CardDescription>
            {`Upload a CSV file with token-level annotations. Your CSV file should follow these requirements:`}
            <br />
            <br />
            {`• Two columns: 'source' and 'target'`}
            <br />
            {`• Source column: Contains full text`}
            <br />
            {`• Target column: Space-separated labels matching each word/token from source`}
            <br />
            {`• IMPORTANT: Number of tokens in source (split by space) MUST match number of labels in target`}
            <br />
            <br />
            {`Example (6 tokens each):`}
            <br />
            {`Source: "The borrower name is John Smith"`}
            <br />
            {`Target: "O O O O NAME NAME"`}
            <br />
            <br />
            <div className="flex flex-wrap gap-2">
              <span className="">Tags Used for Training: </span>
              <div className="w-fit max-w-[600px]">
                {/* Tags Box */}
                <div className="p-1 border-2 border-slate-300 rounded-lg flex flex-wrap">
                  {tags.map(
                    (tag, index) =>
                      index < numTagDisplay && (
                        <div className="rounded-lg p-2 m-2 bg-slate-100" key={`${index}-${tag}`}>
                          {tag}
                        </div>
                      )
                  )}
                  {tags.length > 5 && (
                    <Button
                      color="inherit"
                      variant="outlined"
                      size="medium"
                      onClick={numTagDisplay === 5 ? handleTagDisplayMore : handleTagDisplayLess}
                    >
                      {numTagDisplay === 5 ? 'Expand ▼' : 'Collapse ▲'}
                    </Button>
                  )}
                </div>
              </div>
            </div>
            <br />
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Box sx={{ mb: 4 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Typography variant="h6" component="h3" sx={{ mr: 1 }}>
                  Name Your Updated Model
                </Typography>
                <Tooltip title="Use alphanumeric characters, hyphens, and underscores only. This will be the identifier for your updated model.">
                  <HelpCircle size={20} />
                </Tooltip>
              </Box>
              <TextField
                fullWidth
                id="model-name"
                label="New Model Name"
                variant="outlined"
                value={newModelName}
                onChange={(e) => setNewModelName(e.target.value)}
                placeholder="Enter new model name"
                helperText={warningMessage || 'Example: my-model-v2 or updated_model_123'}
                error={!!warningMessage}
                sx={{ mt: 1 }}
              />
            </Box>
            <div
              className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition-colors"
              onClick={() => document.getElementById('file-input')?.click()}
            >
              <input
                type="file"
                id="file-input"
                className="hidden"
                accept=".csv"
                onChange={handleFileInput}
              />
              <Upload className="mx-auto mb-2 text-gray-400" size={24} />
              {selectedFile ? (
                <p className="text-green-600">Selected: {selectedFile.name}</p>
              ) : (
                <p className="text-gray-600">Click to select a CSV file</p>
              )}
            </div>

            {uploadError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {uploadError}
              </Alert>
            )}

            {uploadSuccess && (
              <Alert severity="success" sx={{ mb: 2 }}>
                Update process initiated successfully with uploaded CSV.
              </Alert>
            )}

            <Button
              onClick={handleUploadUpdate}
              disabled={isUploadUpdating || !selectedFile || uploadButtonDisabled}
              variant="contained"
              color={uploadSuccess ? 'success' : 'primary'}
              fullWidth
            >
              {isUploadUpdating
                ? 'Initiating Update...'
                : uploadSuccess
                  ? 'Update Initiated!'
                  : 'Update Model with CSV'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Polled Data Section with Recent Samples */}
      <Card>
        <CardHeader>
          <CardTitle>Update Model with Recent User Feedback</CardTitle>
          <CardDescription>View and use recent labeled samples to update the model</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <Box sx={{ mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="h6" component="h3" sx={{ mr: 1 }}>
                Name Your New Model
              </Typography>
              <Tooltip title="Use alphanumeric characters, hyphens, and underscores only. This will be the identifier for your updated model.">
                <HelpCircle size={20} />
              </Tooltip>
            </Box>
            <TextField
              fullWidth
              id="polling-model-name"
              label="New Model Name"
              variant="outlined"
              value={pollingModelName}
              onChange={(e) => setPollingModelName(e.target.value)}
              placeholder="Enter new model name"
              helperText={pollingWarningMessage || 'Example: my-model-v2 or updated_model_123'}
              error={!!pollingWarningMessage}
              sx={{ mt: 1 }}
            />
          </Box>

          <div className="mb-6">
            <RecentSamples deploymentUrl={deploymentUrl} />
          </div>

          <div className="space-y-4">
            {pollingError && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {pollingError}
              </Alert>
            )}

            {pollingSuccess && (
              <Alert severity="success" sx={{ mb: 2 }}>
                Update process initiated successfully with polled data.
              </Alert>
            )}

            <Button
              onClick={handlePollingUpdate}
              disabled={isPollingUpdating || pollingButtonDisabled || deployStatus !== 'complete'}
              variant="contained"
              color={pollingSuccess ? 'success' : 'primary'}
              fullWidth
            >
              {isPollingUpdating
                ? 'Initiating Update...'
                : pollingSuccess
                  ? 'Update Initiated!'
                  : deployStatus !== 'complete'
                    ? "Model Must Be Deployed First (refresh page once it's deployed)"
                    : 'Update Model with User Feedback'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelUpdate; 