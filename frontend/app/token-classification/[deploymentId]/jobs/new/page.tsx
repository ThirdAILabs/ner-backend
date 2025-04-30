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
  const [groups, setGroups] = useState<Record<string, string>>({});
  const [newGroupName, setNewGroupName] = useState('');
  const [newGroupQuery, setNewGroupQuery] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const modelsData = await nerService.listModels();
        setModels(modelsData);
      } catch (err) {
        setError('Failed to fetch models');
        console.error('Error fetching models:', err);
      }
    };

    fetchModels();
  }, []);

  const handleAddGroup = () => {
    if (newGroupName && newGroupQuery) {
      setGroups(prev => ({
        ...prev,
        [newGroupName]: newGroupQuery
      }));
      setNewGroupName('');
      setNewGroupQuery('');
    }
  };

  const handleRemoveGroup = (groupName: string) => {
    const newGroups = { ...groups };
    delete newGroups[groupName];
    setGroups(newGroups);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedModelId) {
      setError('Please select a model');
      return;
    }
    
    if (!sourceS3Bucket) {
      setError('S3 bucket is required');
      return;
    }
    
    if (Object.keys(groups).length === 0) {
      setError('At least one group is required');
      return;
    }
    
    setError(null);
    setIsSubmitting(true);
    
    try {
      const response = await nerService.createReport({
        ModelId: selectedModelId,
        SourceS3Bucket: sourceS3Bucket,
        SourceS3Prefix: sourceS3Prefix || undefined,
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
                    label="S3 Prefix (Optional)"
                    variant="outlined"
                    fullWidth
                    value={sourceS3Prefix}
                    onChange={(e) => setSourceS3Prefix(e.target.value)}
                    disabled={isSubmitting}
                    helperText="The prefix within the S3 bucket (optional)"
                  />
                </Box>

                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Groups
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <TextField
                      label="Group Name"
                      variant="outlined"
                      value={newGroupName}
                      onChange={(e) => setNewGroupName(e.target.value)}
                      sx={{ mr: 2, width: '200px' }}
                      disabled={isSubmitting}
                    />
                    <TextField
                      label="Query"
                      variant="outlined"
                      value={newGroupQuery}
                      onChange={(e) => setNewGroupQuery(e.target.value)}
                      sx={{ mr: 2, width: '300px' }}
                      disabled={isSubmitting}
                    />
                    <Button
                      variant="outlined"
                      onClick={handleAddGroup}
                      disabled={isSubmitting || !newGroupName || !newGroupQuery}
                    >
                      Add Group
                    </Button>
                  </Box>

                  {Object.entries(groups).map(([name, query]) => (
                    <Box key={name} sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
                      <Typography sx={{ mr: 2, fontWeight: 'bold' }}>{name}:</Typography>
                      <Typography sx={{ flex: 1 }}>{query}</Typography>
                      <Button
                        color="error"
                        onClick={() => handleRemoveGroup(name)}
                        disabled={isSubmitting}
                      >
                        Remove
                      </Button>
                    </Box>
                  ))}
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