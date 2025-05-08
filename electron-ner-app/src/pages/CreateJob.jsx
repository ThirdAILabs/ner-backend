import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  FormControl,
  FormLabel,
  RadioGroup,
  Radio,
  FormControlLabel,
  Paper,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { nerService } from '../lib/backend';

// Source option component
const SourceOption = ({ title, description, isSelected, disabled, onClick }) => (
  <Paper
    elevation={0}
    sx={{
      p: 3,
      border: '1px solid',
      borderColor: isSelected ? 'primary.main' : 'divider',
      borderRadius: 1,
      opacity: disabled ? 0.5 : 1,
      cursor: disabled ? 'not-allowed' : 'pointer',
      position: 'relative',
      '&:hover': !disabled && {
        borderColor: isSelected ? 'primary.main' : 'primary.light',
      },
      '&::before': isSelected ? {
        content: '""',
        position: 'absolute',
        right: 10,
        top: 10,
        width: 10,
        height: 10,
        borderRadius: '50%',
        bgcolor: 'primary.main'
      } : {}
    }}
    onClick={() => !disabled && onClick()}
  >
    <Typography variant="subtitle1" fontWeight="medium" gutterBottom>
      {title}
    </Typography>
    <Typography variant="body2" color="text.secondary">
      {description}
    </Typography>
  </Paper>
);

// Tag component
const Tag = ({ tag, selected, onClick }) => (
  <Chip
    label={tag}
    clickable
    onClick={onClick}
    sx={{
      bgcolor: selected ? 'primary.main' : 'grey.100',
      color: selected ? 'white' : 'text.primary',
      '&:hover': {
        bgcolor: selected ? 'primary.dark' : 'grey.200',
      },
      m: 0.5
    }}
  />
);

// Custom tag dialog
const CustomTagDialog = ({ open, onClose, onAdd, existingTags }) => {
  const [tagName, setTagName] = useState('');
  const [pattern, setPattern] = useState('');
  const [error, setError] = useState('');
  
  const handleSubmit = () => {
    if (!tagName.trim() || !pattern.trim()) {
      setError('Both fields are required');
      return;
    }
    
    if (existingTags.includes(tagName.toUpperCase())) {
      setError('Tag name already exists');
      return;
    }
    
    onAdd({ name: tagName.toUpperCase(), pattern });
    setTagName('');
    setPattern('');
    setError('');
    onClose();
  };
  
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Create Custom Tag</DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <TextField
          autoFocus
          margin="dense"
          label="Tag Name"
          fullWidth
          value={tagName}
          onChange={(e) => setTagName(e.target.value.toUpperCase())}
          sx={{ mb: 2 }}
        />
        <TextField
          margin="dense"
          label="Regex Pattern"
          fullWidth
          value={pattern}
          onChange={(e) => setPattern(e.target.value)}
          helperText="Example: \d{3}[-.]?\d{3}[-.]?\d{4} for phone numbers"
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained">Add</Button>
      </DialogActions>
    </Dialog>
  );
};

// Group component
const GroupCard = ({ name, definition, onRemove }) => (
  <Paper elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: 1, overflow: 'hidden' }}>
    <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid', borderColor: 'divider' }}>
      <Typography variant="subtitle1" fontWeight="medium">
        {name}
      </Typography>
      <Button variant="text" color="error" size="small" onClick={onRemove}>
        Remove
      </Button>
    </Box>
    <Box sx={{ p: 2 }}>
      <Typography variant="body2" fontFamily="monospace">
        {definition}
      </Typography>
    </Box>
  </Paper>
);

// Main CreateJob component
const CreateJob = ({ onBack }) => {
  // Essential state
  const [selectedSource, setSelectedSource] = useState('s3');
  const [sourceS3Bucket, setSourceS3Bucket] = useState('');
  const [sourceS3Prefix, setSourceS3Prefix] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Job Name
  const [jobName, setJobName] = useState('');

  // Model selection
  const [models, setModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState('');

  // Tags handling
  const [availableTags, setAvailableTags] = useState([]);
  const [selectedTags, setSelectedTags] = useState([]);
  const [isTagsLoading, setIsTagsLoading] = useState(false);

  // Groups handling
  const [groupName, setGroupName] = useState('');
  const [groupQuery, setGroupQuery] = useState('');
  const [groups, setGroups] = useState({});

  // Custom tags handling
  const [customTags, setCustomTags] = useState([]);
  const [isCustomTagDialogOpen, setIsCustomTagDialogOpen] = useState(false);

  // Status
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  // Fetch models when component mounts
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const modelData = await nerService.listModels();
        // Only show trained models
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
        
        // Get tags from the model
        const modelTags = model.Tags || [];
        
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
  const toggleTag = (tag) => {
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

  // File selection handler
  const handleFileChange = (e) => {
    const files = e.target.files;
    if (files) {
      setSelectedFiles(Array.from(files));
    }
  };

  // Add a new group
  const handleAddGroup = () => {
    if (!groupName || !groupQuery) {
      setError('Group name and query are required');
      return;
    }

    setGroups({
      ...groups,
      [groupName]: groupQuery
    });

    setGroupName('');
    setGroupQuery('');
  };

  // Remove a group
  const handleRemoveGroup = (name) => {
    const newGroups = { ...groups };
    delete newGroups[name];
    setGroups(newGroups);
  };

  // Add a custom tag
  const handleAddCustomTag = (customTag) => {
    setCustomTags([...customTags, customTag]);
    setAvailableTags([...availableTags, customTag.name]);
    setSelectedTags([...selectedTags, customTag.name]);
  };

  // Submit handler
  const handleSubmit = async (e) => {
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
      const customTagsObj = {};
      customTags.forEach(tag => {
        customTagsObj[tag.name] = tag.pattern;
      });

      // Create the report
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
      
      // Go back to job list after success
      setTimeout(() => {
        onBack(response.ReportId);
      }, 2000);
    } catch (err) {
      setError('Failed to create report. Please try again.');
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Box sx={{ p: 3, bgcolor: 'white', borderRadius: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" fontWeight="medium">Create New Job</Typography>
        <Button 
          variant="outlined" 
          startIcon={<ArrowBackIcon />} 
          onClick={onBack}
        >
          Back to Jobs
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {success ? (
        <Alert severity="success" sx={{ mb: 3 }}>
          Job created successfully! Redirecting...
        </Alert>
      ) : (
        <form onSubmit={handleSubmit}>
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>Job Name</Typography>
            <TextField
              fullWidth
              value={jobName}
              onChange={(e) => {
                // Replace spaces with underscores
                const value = e.target.value.replace(/\s/g, '_');
                setJobName(value);
              }}
              placeholder="my_job_name"
              helperText="Use only letters, numbers, and underscores. No spaces allowed."
              required
              sx={{ maxWidth: 'md' }}
            />
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>Model</Typography>
            <Grid container spacing={2}>
              {models.map(model => (
                <Grid item xs={12} md={4} key={model.Id}>
                  <SourceOption
                    title={model.Name}
                    description={model.Description || 'No description available'}
                    isSelected={selectedModelId === model.Id}
                    onClick={() => setSelectedModelId(model.Id)}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>Source</Typography>
            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid item xs={12} md={6}>
                <SourceOption
                  title="S3 Bucket"
                  description="Use files from an S3 bucket"
                  isSelected={selectedSource === 's3'}
                  onClick={() => setSelectedSource('s3')}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <SourceOption
                  title="File Upload"
                  description="Upload files from your computer"
                  isSelected={selectedSource === 'files'}
                  onClick={() => setSelectedSource('files')}
                />
              </Grid>
            </Grid>

            {selectedSource === 's3' ? (
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="S3 Bucket Name"
                    value={sourceS3Bucket}
                    onChange={(e) => setSourceS3Bucket(e.target.value)}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="S3 Prefix (Optional)"
                    value={sourceS3Prefix}
                    onChange={(e) => setSourceS3Prefix(e.target.value)}
                    placeholder="folder/path/"
                  />
                </Grid>
              </Grid>
            ) : (
              <Box>
                <input
                  accept=".pdf,.txt,.csv,.html,.json,.xml"
                  style={{ display: 'none' }}
                  id="raised-button-file"
                  multiple
                  type="file"
                  onChange={handleFileChange}
                />
                <label htmlFor="raised-button-file">
                  <Button 
                    variant="outlined" 
                    component="span"
                    fullWidth
                    sx={{ 
                      height: '100px', 
                      display: 'flex', 
                      flexDirection: 'column',
                      justifyContent: 'center',
                      border: '2px dashed',
                      borderColor: 'divider'
                    }}
                  >
                    <Typography>Select files</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Drag and drop files or click to browse
                    </Typography>
                  </Button>
                </label>

                {selectedFiles.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Selected Files ({selectedFiles.length})
                    </Typography>
                    <Paper sx={{ maxHeight: 200, overflow: 'auto', p: 1 }}>
                      {selectedFiles.map((file, index) => (
                        <Box 
                          key={index}
                          sx={{ 
                            display: 'flex', 
                            justifyContent: 'space-between',
                            p: 1,
                            borderBottom: index < selectedFiles.length - 1 ? '1px solid' : 'none',
                            borderColor: 'divider'
                          }}
                        >
                          <Typography variant="body2">
                            {file.name} ({(file.size / 1024).toFixed(1)} KB)
                          </Typography>
                          <Button 
                            size="small"
                            color="error"
                            onClick={() => {
                              const newFiles = [...selectedFiles];
                              newFiles.splice(index, 1);
                              setSelectedFiles(newFiles);
                            }}
                          >
                            Remove
                          </Button>
                        </Box>
                      ))}
                    </Paper>
                  </Box>
                )}
              </Box>
            )}
          </Box>

          {selectedModelId && (
            <Box sx={{ mb: 4 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Tags</Typography>
                <Button 
                  variant="outlined" 
                  size="small"
                  onClick={selectAllTags}
                  disabled={isTagsLoading || selectedTags.length === availableTags.length}
                >
                  Select All
                </Button>
              </Box>

              {isTagsLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                  <CircularProgress size={24} />
                </Box>
              ) : availableTags.length === 0 ? (
                <Typography color="text.secondary">No tags available for this model</Typography>
              ) : (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {availableTags.map(tag => (
                    <Tag
                      key={tag}
                      tag={tag}
                      selected={selectedTags.includes(tag)}
                      onClick={() => toggleTag(tag)}
                    />
                  ))}
                </Box>
              )}
            </Box>
          )}

          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>Custom Tags</Typography>
            
            <Grid container spacing={2}>
              {customTags.map((tag, index) => (
                <Grid item xs={12} md={4} key={index}>
                  <GroupCard
                    name={tag.name}
                    definition={tag.pattern}
                    onRemove={() => {
                      setCustomTags(customTags.filter((_, i) => i !== index));
                      setAvailableTags(availableTags.filter(t => t !== tag.name));
                      setSelectedTags(selectedTags.filter(t => t !== tag.name));
                    }}
                  />
                </Grid>
              ))}
              
              <Grid item xs={12} md={4}>
                <Paper
                  elevation={0}
                  sx={{
                    height: '100%',
                    minHeight: '100px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '2px dashed',
                    borderColor: 'divider',
                    borderRadius: 1,
                    cursor: 'pointer',
                    '&:hover': {
                      borderColor: 'primary.light',
                      bgcolor: 'action.hover',
                    }
                  }}
                  onClick={() => setIsCustomTagDialogOpen(true)}
                >
                  <Typography color="text.secondary">
                    + Add Custom Tag
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
            
            <CustomTagDialog
              open={isCustomTagDialogOpen}
              onClose={() => setIsCustomTagDialogOpen(false)}
              onAdd={handleAddCustomTag}
              existingTags={availableTags}
            />
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>Groups</Typography>
            
            <Card sx={{ mb: 3, p: 2 }}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Group Name"
                    value={groupName}
                    onChange={(e) => setGroupName(e.target.value)}
                    placeholder="sensitive_docs"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Group Query"
                    value={groupQuery}
                    onChange={(e) => setGroupQuery(e.target.value)}
                    placeholder="COUNT(SSN) > 0"
                  />
                </Grid>
              </Grid>
              
              <Box sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  onClick={handleAddGroup}
                  disabled={!groupName || !groupQuery}
                >
                  Add Group
                </Button>
              </Box>
              
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Example queries:
                </Typography>
                <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
                  <li>
                    <Typography variant="body2" fontFamily="monospace">
                      COUNT(SSN) &gt; 0
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Documents containing SSNs
                    </Typography>
                  </li>
                  <li>
                    <Typography variant="body2" fontFamily="monospace">
                      COUNT(NAME) &gt; 2 AND COUNT(PHONE) &gt; 0
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Documents with multiple names and a phone number
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Card>
            
            <Grid container spacing={2}>
              {Object.entries(groups).map(([name, query]) => (
                <Grid item xs={12} md={6} key={name}>
                  <GroupCard
                    name={name}
                    definition={query}
                    onRemove={() => handleRemoveGroup(name)}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'flex-end', pt: 2 }}>
            <Button
              type="submit"
              variant="contained"
              disabled={isSubmitting}
              size="large"
            >
              {isSubmitting ? (
                <>
                  <CircularProgress size={20} sx={{ mr: 1 }} />
                  Creating...
                </>
              ) : (
                'Create Job'
              )}
            </Button>
          </Box>
        </form>
      )}
    </Box>
  );
};

export default CreateJob; 