'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import {
  Box,
  Button,
  Typography,
  TextField,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormHelperText,
  CircularProgress,
  Alert,
  Breadcrumbs,
  Link as MuiLink,
} from '@mui/material';
import { nerService } from '@/lib/backend';

interface Model {
  Id: string;
  Name: string;
  Type: string;
  Status: string;
}

export default function NewJobPage() {
  const params = useParams();
  const router = useRouter();
  const deploymentId = params.deploymentId as string;
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [sourceS3Bucket, setSourceS3Bucket] = useState('');
  const [sourceS3Prefix, setSourceS3Prefix] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadMode, setUploadMode] = useState<'s3' | 'files'>('s3');
  const [groupName, setGroupName] = useState('');
  const [groupQuery, setGroupQuery] = useState('');
  const [groups, setGroups] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const modelData = await nerService.listModels();
        // Only show trained models that can be used for inference
        const trainedModels = modelData.filter(model => model.Status === 'TRAINED');
        setModels(trainedModels);
      } catch (err) {
        console.error('Error fetching models:', err);
      }
    };

    fetchModels();
  }, []);

  const handleAddGroup = () => {
    if (!groupName.trim()) {
      setError('Group name is required');
      return;
    }

    if (!groupQuery.trim()) {
      setError('Group query is required');
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

  const handleRemoveGroup = (name: string) => {
    const newGroups = { ...groups };
    delete newGroups[name];
    setGroups(newGroups);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setSelectedFiles(Array.from(e.target.files));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedModelId) {
      setError('Please select a model');
      return;
    }
    
    if (uploadMode === 's3' && !sourceS3Bucket) {
      setError('S3 bucket is required');
      return;
    }
    
    if (uploadMode === 'files' && selectedFiles.length === 0) {
      setError('Please select at least one file');
      return;
    }
    
    if (Object.keys(groups).length === 0) {
      setError('At least one group is required');
      return;
    }
    
    setError(null);
    setIsSubmitting(true);
    
    try {
      let uploadId;
      
      // Handle file uploads if needed
      if (uploadMode === 'files') {
        const uploadResponse = await nerService.uploadFiles(selectedFiles);
        uploadId = uploadResponse.Id;
      }
      
      // Create the report
      const response = await nerService.createReport({
        ModelId: selectedModelId,
        ...(uploadMode === 's3' ? {
          SourceS3Bucket: sourceS3Bucket,
          SourceS3Prefix: sourceS3Prefix || undefined,
        } : {
          UploadId: uploadId,
          SourceS3Bucket: 'uploads', // This value doesn't matter when UploadId is set
        }),
        Groups: groups
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
    <div className="bg-muted min-h-screen">
      <header className="w-full p-4 bg-muted border-b">
        <div className="max-w-7xl mx-auto space-y-4">
          <Breadcrumbs>
            <MuiLink href="#" underline="hover" color="inherit">
              Token Classification
            </MuiLink>
            <MuiLink 
              href={`/token-classification/${deploymentId}`} 
              underline="hover" 
              color="inherit"
            >
              {deploymentId}
            </MuiLink>
            <MuiLink 
              href={`/token-classification/${deploymentId}?tab=jobs`} 
              underline="hover" 
              color="inherit"
            >
              Jobs
            </MuiLink>
            <Typography color="text.primary">New Report</Typography>
          </Breadcrumbs>
          
          <Typography variant="h5" className="font-bold">
            Create New Report
          </Typography>
        </div>
      </header>

      <main className="max-w-3xl mx-auto p-4">
        <Card>
          <CardContent>
            {success ? (
              <Alert severity="success" sx={{ mb: 2 }}>
                Report created successfully! Redirecting...
              </Alert>
            ) : (
              <form onSubmit={handleSubmit}>
                {error && (
                  <Alert severity="error" sx={{ mb: 3 }}>
                    {error}
                  </Alert>
                )}
                
                <Box sx={{ mb: 3 }}>
                  <FormControl fullWidth required>
                    <InputLabel>Model</InputLabel>
                    <Select
                      value={selectedModelId}
                      onChange={(e) => setSelectedModelId(e.target.value)}
                      label="Model"
                      disabled={isSubmitting}
                    >
                      {models.map((model) => (
                        <MenuItem key={model.Id} value={model.Id}>
                          {model.Name} ({model.Type})
                        </MenuItem>
                      ))}
                    </Select>
                    <FormHelperText>
                      Select the model to use for this report
                    </FormHelperText>
                  </FormControl>
                </Box>

                {/* Source Selection Tabs */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Data Source
                  </Typography>
                  <Box sx={{ display: 'flex', mb: 2 }}>
                    <Button 
                      variant={uploadMode === 's3' ? 'contained' : 'outlined'}
                      onClick={() => setUploadMode('s3')}
                      sx={{ mr: 1 }}
                    >
                      S3 Bucket
                    </Button>
                    <Button 
                      variant={uploadMode === 'files' ? 'contained' : 'outlined'}
                      onClick={() => setUploadMode('files')}
                    >
                      Upload Files
                    </Button>
                  </Box>

                  {uploadMode === 's3' ? (
                    <>
                      <Box sx={{ mb: 3 }}>
                        <TextField
                          label="S3 Bucket"
                          variant="outlined"
                          fullWidth
                          value={sourceS3Bucket}
                          onChange={(e) => setSourceS3Bucket(e.target.value)}
                          required
                          disabled={isSubmitting}
                          helperText="The S3 bucket containing the source data"
                        />
                      </Box>

                      <Box sx={{ mb: 3 }}>
                        <TextField
                          label="S3 Prefix"
                          variant="outlined"
                          fullWidth
                          value={sourceS3Prefix}
                          onChange={(e) => setSourceS3Prefix(e.target.value)}
                          disabled={isSubmitting}
                          helperText="Optional: S3 prefix filter (folder path)"
                        />
                      </Box>
                    </>
                  ) : (
                    <Box sx={{ mb: 3 }}>
                      <input
                        type="file"
                        multiple
                        onChange={handleFileChange}
                        disabled={isSubmitting}
                        style={{ display: 'none' }}
                        id="file-upload-input"
                      />
                      <label htmlFor="file-upload-input">
                        <Button
                          variant="outlined"
                          component="span"
                          disabled={isSubmitting}
                          fullWidth
                          sx={{ py: 2, borderStyle: 'dashed' }}
                        >
                          Click to select files
                        </Button>
                      </label>
                      
                      {selectedFiles.length > 0 && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Selected Files ({selectedFiles.length}):
                          </Typography>
                          <Box sx={{ maxHeight: 150, overflow: 'auto', border: '1px solid #ddd', p: 1, borderRadius: 1 }}>
                            {selectedFiles.map((file, index) => (
                              <Typography key={index} variant="body2" sx={{ mb: 0.5 }}>
                                {file.name} ({(file.size / 1024).toFixed(1)} KB)
                              </Typography>
                            ))}
                          </Box>
                        </Box>
                      )}
                    </Box>
                  )}
                </Box>

                <Box sx={{ mb: 4 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Groups
                  </Typography>
                  <Box sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                    <Typography variant="body2" gutterBottom>
                      Define groups to categorize your data. Examples:
                    </Typography>
                    <ul style={{ paddingLeft: 20, fontSize: '0.875rem' }}>
                      <li><code>COUNT(SSN) {'>'} 0</code> - Documents containing SSNs</li>
                      <li><code>COUNT(NAME) {'>'} 2 AND COUNT(PHONE) {'>'} 0</code> - Documents with multiple names and a phone number</li>
                      <li><code>EMAIL CONTAINS "gmail.com"</code> - Documents with Gmail addresses</li>
                    </ul>
                  </Box>
                  
                  <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
                    <TextField
                      label="Group Name"
                      variant="outlined"
                      value={groupName}
                      onChange={(e) => setGroupName(e.target.value)}
                      disabled={isSubmitting}
                      sx={{ flex: 1 }}
                    />
                    <TextField
                      label="Group Query"
                      variant="outlined"
                      value={groupQuery}
                      onChange={(e) => setGroupQuery(e.target.value)}
                      disabled={isSubmitting}
                      sx={{ flex: 2 }}
                    />
                    <Button
                      variant="outlined"
                      onClick={handleAddGroup}
                      disabled={isSubmitting || !groupName || !groupQuery}
                    >
                      Add
                    </Button>
                  </Box>
                  
                  <Box sx={{ mb: 3 }}>
                    {Object.entries(groups).map(([name, query]) => (
                      <Box key={name} sx={{ mb: 1, display: 'flex', alignItems: 'center', p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        <Typography sx={{ mr: 2, fontWeight: 'bold' }}>{name}:</Typography>
                        <Typography sx={{ flex: 1 }}>{query}</Typography>
                        <Button
                          color="error"
                          onClick={() => handleRemoveGroup(name)}
                          disabled={isSubmitting}
                          size="small"
                        >
                          Remove
                        </Button>
                      </Box>
                    ))}
                  </Box>
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Button
                    variant="outlined"
                    onClick={() => router.push(`/token-classification/${deploymentId}`)}
                    disabled={isSubmitting}
                  >
                    Cancel
                  </Button>
                  
                  <Button
                    type="submit"
                    variant="contained"
                    disabled={isSubmitting}
                    startIcon={isSubmitting ? <CircularProgress size={20} /> : null}
                  >
                    {isSubmitting ? 'Creating...' : 'Create Report'}
                  </Button>
                </Box>
              </form>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
} 