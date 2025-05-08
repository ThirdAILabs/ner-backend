import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Card,
  CardContent,
  Divider,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
  TextField,
  InputAdornment,
  IconButton,
} from '@mui/material';
import { format } from 'date-fns';
import { nerService } from '../lib/backend';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import SearchIcon from '@mui/icons-material/Search';
import SaveIcon from '@mui/icons-material/Save';

// Configuration tab component
const ConfigurationTab = ({ report }) => {
  if (!report) return null;
  
  // Determine if it's a file upload or S3 bucket
  const isFileUpload = report.IsUpload === true;
  const isS3 = !isFileUpload;
  
  return (
    <Box sx={{ mt: 3 }}>
      <div className="space-y-8">
        {/* Source section */}
        <div>
          <Typography 
            variant="h6" 
            sx={{ 
              fontSize: '1.125rem', 
              fontWeight: 500, 
              mb: 2 
            }}
          >
            Source
          </Typography>
          <Box 
            sx={{ 
              display: 'grid', 
              gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' }, 
              gap: 2 
            }}
          >
            {/* S3 Bucket Option */}
            <Paper 
              elevation={0} 
              sx={{ 
                p: 2, 
                border: '1px solid',
                borderColor: isS3 ? 'primary.main' : 'divider',
                borderRadius: 1,
                position: 'relative',
                opacity: isS3 ? 1 : 0.5,
                '&::before': isS3 ? {
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
            >
              <Typography fontWeight="medium" gutterBottom>S3 Bucket</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ wordBreak: 'break-all' }}>
                {report.SourceS3Bucket}/{report.SourceS3Prefix || ''}
              </Typography>
            </Paper>
            
            {/* File Upload Option */}
            <Paper 
              elevation={0} 
              sx={{ 
                p: 2, 
                border: '1px solid',
                borderColor: isFileUpload ? 'primary.main' : 'divider',
                borderRadius: 1,
                opacity: isFileUpload ? 1 : 0.5,
                position: 'relative',
                '&::before': isFileUpload ? {
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
            >
              <Typography fontWeight="medium" gutterBottom>File Upload</Typography>
              <Typography variant="body2" color="text.secondary">
                {isFileUpload ? "Files were uploaded directly" : ""}
              </Typography>
            </Paper>
          </Box>
        </div>

        {/* Tags section */}
        <div>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography 
              variant="h6" 
              sx={{ 
                fontSize: '1.125rem', 
                fontWeight: 500 
              }}
            >
              Tags
            </Typography>
          </Box>

          {report.Tags && report.Tags.length > 0 ? (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {report.Tags.map((tag, index) => (
                <Chip 
                  key={index} 
                  label={tag} 
                  variant="filled"
                  sx={{ 
                    bgcolor: 'rgba(25, 118, 210, 0.08)',
                    color: 'primary.main',
                    fontWeight: 500,
                    borderRadius: '4px',
                    '&:hover': {
                      bgcolor: 'rgba(25, 118, 210, 0.12)',
                    }
                  }} 
                />
              ))}
            </Box>
          ) : (
            <Typography color="text.secondary" sx={{ py: 1 }}>
              No tags available
            </Typography>
          )}
        </div>
      </div>
    </Box>
  );
};

// Analytics tab component
const AnalyticsTab = ({ report }) => {
  if (!report) return null;
  
  // Calculate progress percentage
  const calculateProgress = () => {
    if (!report.InferenceTaskStatuses) return 0;
    
    const completed = report.InferenceTaskStatuses?.COMPLETED?.TotalTasks || 0;
    const running = report.InferenceTaskStatuses?.RUNNING?.TotalTasks || 0;
    const queued = report.InferenceTaskStatuses?.QUEUED?.TotalTasks || 0;
    const failed = report.InferenceTaskStatuses?.FAILED?.TotalTasks || 0;
    
    const totalTasks = completed + running + queued + failed;
    return totalTasks > 0 ? Math.round((completed / totalTasks) * 100) : 0;
  };
  
  // Get processed tokens
  const getProcessedTokens = () => {
    // This is actually bytes processed, not tokens
    if (!report || !report.InferenceTaskStatuses || !report.InferenceTaskStatuses.COMPLETED) {
      return 0;
    }
    
    return report.InferenceTaskStatuses.COMPLETED.TotalSize || 0;
  };
  
  // Convert tag counts to format expected by AnalyticsDashboard
  const formatTagCounts = () => {
    if (!report.TagCounts) return [];
    
    return Object.entries(report.TagCounts).map(([type, count]) => ({
      type,
      count
    }));
  };
  
  return (
    <Box sx={{ mt: 3 }}>
      <AnalyticsDashboard
        progress={calculateProgress()}
        tokensProcessed={getProcessedTokens()}
        tags={formatTagCounts()}
      />
    </Box>
  );
};

// Output tab component
const OutputTab = ({ entities, reportId }) => {
  const [viewMode, setViewMode] = useState('object');
  const [objectPreviews, setObjectPreviews] = useState([]);
  const [loadingObjectData, setLoadingObjectData] = useState(false);
  const [classifiedTokens, setClassifiedTokens] = useState(entities);
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState(null);
  
  // Pagination states
  const [loadingMoreTokens, setLoadingMoreTokens] = useState(false);
  const [tokenOffset, setTokenOffset] = useState(0);
  const [hasMoreTokens, setHasMoreTokens] = useState(true);
  const TOKENS_LIMIT = 20;

  // Pagination for object view
  const [loadingMoreObjects, setLoadingMoreObjects] = useState(false);
  const [objectOffset, setObjectOffset] = useState(0);
  const [hasMoreObjects, setHasMoreObjects] = useState(true);
  const OBJECTS_LIMIT = 10;

  // Debug logs
  console.log('OutputTab render - entities received:', entities?.length);
  console.log('OutputTab render - reportId:', reportId);
  console.log('OutputTab render - classifiedTokens length:', classifiedTokens?.length);
  console.log('OutputTab render - objectPreviews length:', objectPreviews?.length);

  // Set initial token records
  useEffect(() => {
    if (entities?.length > 0) {
      setClassifiedTokens(entities);
      setTokenOffset(entities.length);
      setHasMoreTokens(entities.length >= TOKENS_LIMIT);
    }
  }, [entities]);

  // Fetch object previews when viewMode changes to 'object'
  useEffect(() => {
    if (!reportId) {
      console.error('OutputTab: No reportId provided');
      setError('No report ID provided');
      return;
    }

    if (viewMode === 'object') {
      console.log('Fetching object previews for reportId:', reportId);
      fetchObjectPreviews();
    }
  }, [viewMode, reportId]); // Include reportId in dependencies to re-fetch if it changes

  // Fetch object data
  const fetchObjectPreviews = async (isLoadingMore = false) => {
    if (isLoadingMore) {
      setLoadingMoreObjects(true);
    } else {
      setLoadingObjectData(true);
      setObjectPreviews([]);
      setObjectOffset(0);
    }
    
    setError(null);
    
    try {
      const offset = isLoadingMore ? objectOffset : 0;
      console.log(`Calling nerService.getReportObjects with reportId: ${reportId}, offset: ${offset}`);
      
      const data = await nerService.getReportObjects(reportId, { 
        offset: offset,
        limit: OBJECTS_LIMIT 
      });
      
      console.log('getReportObjects response:', data?.length || 0, 'objects');
      
      if (data && Array.isArray(data)) {
        if (isLoadingMore) {
          setObjectPreviews(prev => [...prev, ...data]);
        } else {
          setObjectPreviews(data);
        }
        
        setObjectOffset(isLoadingMore ? objectOffset + data.length : data.length);
        setHasMoreObjects(data.length >= OBJECTS_LIMIT);
        
        console.log('Object previews updated, length:', isLoadingMore ? 
          objectPreviews.length + data.length : data.length);
      } else {
        console.error('Invalid data returned from getReportObjects', data);
        setError('Invalid data format returned from API');
        setHasMoreObjects(false);
      }
    } catch (error) {
      console.error('Error fetching object previews:', error);
      setError(`Error fetching object data: ${error.message}`);
      setHasMoreObjects(false);
    } finally {
      if (isLoadingMore) {
        setLoadingMoreObjects(false);
      } else {
        setLoadingObjectData(false);
      }
    }
  };
  
  // Load more objects
  const loadMoreObjects = () => {
    fetchObjectPreviews(true);
  };

  // Load more token records
  const loadMoreTokens = async () => {
    if (loadingMoreTokens || !hasMoreTokens) return;
    
    setLoadingMoreTokens(true);
    try {
      console.log(`Loading more tokens from offset ${tokenOffset}`);
      const moreEntities = await nerService.getReportEntities(reportId, { 
        offset: tokenOffset, 
        limit: TOKENS_LIMIT 
      });
      
      if (moreEntities && Array.isArray(moreEntities)) {
        console.log(`Loaded ${moreEntities.length} more tokens`);
        setClassifiedTokens(prev => [...prev, ...moreEntities]);
        setTokenOffset(prev => prev + moreEntities.length);
        setHasMoreTokens(moreEntities.length >= TOKENS_LIMIT);
      } else {
        console.error('Invalid data returned when loading more tokens');
        setHasMoreTokens(false);
      }
    } catch (error) {
      console.error('Error loading more tokens:', error);
      setError(`Error loading more data: ${error.message}`);
    } finally {
      setLoadingMoreTokens(false);
    }
  };

  const handleViewModeChange = (event, newValue) => {
    setViewMode(newValue);
  };

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setSearching(true);
    try {
      // Implement search functionality here
      console.log("Searching for:", query);
      // Example API call: 
      // const results = await nerService.searchReport(reportId, query);
    } catch (error) {
      console.error("Error searching:", error);
    } finally {
      setSearching(false);
    }
  };

  const handleSave = () => {
    console.log("Saving current view");
    // Implement save functionality
  };

  // Define color palettes for tag colors
  const PASTELS = ['#E5A49C', '#F6C886', '#FBE7AA', '#99E3B5', '#A6E6E7', '#A5A1E1', '#D8A4E2'];
  const DARKERS = ['#D34F3E', '#F09336', '#F7CF5F', '#5CC96E', '#65CFD0', '#597CE2', '#B64DC8'];

  // Generate tag colors
  const getTagColors = () => {
    // Default color for unknown tags
    const DEFAULT_COLOR = { text: '#E0E0E0', tag: '#A0A0A0' };
    
    const colors = {};
    
    // Get all unique tags from entities
    const uniqueTags = [...new Set(entities.map(entity => entity.Label))].filter(tag => tag !== 'O');
    
    // Assign colors to each tag
    uniqueTags.forEach((tag, index) => {
      colors[tag] = {
        text: PASTELS[index % PASTELS.length],
        tag: DARKERS[index % DARKERS.length]
      };
    });
    
    return colors;
  };
  
  // Memoize tag colors so they don't re-compute on every render
  const tagColors = React.useMemo(() => getTagColors(), [entities]);

  // Helper component for token highlighting
  const HighlightedToken = ({ token, tag }) => {
    // Check if token needs space before or after based on its content
    const needsSpaceBefore = !(
      token.match(/^[.,;:!?)\]}"'%]/) || // Don't add space before punctuation
      token.trim() === ''                // Don't add space before empty tokens
    );
    
    const needsSpaceAfter = !(
      token.match(/^[([{"'$]/) ||       // Don't add space after opening brackets
      token.match(/[.,;:!?]$/) ||       // Don't add space after punctuation at the end
      token.trim() === ''                // Don't add space after empty tokens
    );

    // If tag is O (outside entity), render as plain text with appropriate spacing
    if (tag === 'O') {
      return (
        <span>
          {needsSpaceBefore && token !== '' && ' '}
          {token}
          {needsSpaceAfter && ' '}
        </span>
      );
    }
    
    // Get color for this tag
    const color = tagColors[tag] || { text: '#E0E0E0', tag: '#A0A0A0' };
    
    return (
      <span>
        {needsSpaceBefore && ' '}
        <span
          style={{
            backgroundColor: color.text,
            padding: '2px 4px',
            borderRadius: '2px',
            userSelect: 'none',
            display: 'inline-flex',
            alignItems: 'center',
            wordBreak: 'break-word'
          }}
        >
          {token}
          <span
            style={{
              backgroundColor: color.tag,
              color: 'white',
              fontSize: '11px',
              fontWeight: 'bold',
              borderRadius: '2px',
              marginLeft: '4px',
              padding: '1px 3px',
            }}
          >
            {tag}
          </span>
        </span>
        {needsSpaceAfter && ' '}
      </span>
    );
  };

  // Load More button component
  const LoadMoreButton = ({ onClick, isLoading }) => {
    return (
      <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
        <Button 
          variant="outlined"
          onClick={onClick}
          disabled={isLoading}
          sx={{ width: '100%', maxWidth: '300px' }}
        >
          {isLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <CircularProgress size={16} sx={{ mr: 1 }} />
              Loading more...
            </Box>
          ) : (
            'Load More'
          )}
        </Button>
      </Box>
    );
  };

  return (
    <Box sx={{ mt: 3 }}>
      {/* View mode selector and search */}
      <Box sx={{ mb: 3, p: 2, bgcolor: '#f5f7fa', borderRadius: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body1" fontWeight="medium">View By</Typography>
          <Tabs
            value={viewMode}
            onChange={handleViewModeChange}
            aria-label="view mode tabs"
            sx={{
              minHeight: '40px',
              '& .MuiTabs-indicator': {
                backgroundColor: 'primary.main'
              }
            }}
          >
            <Tab 
              label="Object" 
              value="object" 
              sx={{ 
                textTransform: 'none',
                minHeight: '40px',
                padding: '0 16px'
              }}
            />
            <Tab 
              label="Classified Token" 
              value="classified-token" 
              sx={{ 
                textTransform: 'none',
                minHeight: '40px',
                padding: '0 16px'
              }}
            />
          </Tabs>

          <Typography variant="body1" fontWeight="medium" sx={{ ml: 2 }}>Query</Typography>
          <Box sx={{ display: 'flex', flex: 1, gap: 1 }}>
            <TextField
              size="small"
              value={query}
              onChange={handleQueryChange}
              placeholder="Enter query..."
              fullWidth
              variant="outlined"
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      edge="end"
                      onClick={handleSearch}
                      disabled={searching}
                      size="small"
                    >
                      {searching ? <CircularProgress size={20} /> : <SearchIcon />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            <IconButton 
              onClick={handleSave}
              color="primary"
              sx={{ 
                width: '40px', 
                height: '40px', 
                bgcolor: 'background.paper',
                border: '1px solid',
                borderColor: 'divider'
              }}
            >
              <SaveIcon fontSize="small" />
            </IconButton>
          </Box>
        </Box>
      </Box>

      {/* Content based on view mode */}
      {viewMode === 'object' ? (
        // Object view
        <>
          <Typography variant="h6" mb={2}>Sample Objects</Typography>
          
          {loadingObjectData ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress size={30} />
            </Box>
          ) : error ? (
            <Typography color="error" sx={{ p: 2 }}>{error}</Typography>
          ) : objectPreviews.length > 0 ? (
            <>
              <TableContainer component={Paper} sx={{ maxHeight: 500 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Full Text with Tagged Tokens</TableCell>
                      <TableCell>Source Object</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {objectPreviews.map((objPreview, idx) => (
                      <TableRow key={idx}>
                        <TableCell sx={{ 
                          maxWidth: '60%',
                          whiteSpace: 'normal',
                          wordBreak: 'break-word',
                          overflowWrap: 'break-word',
                          padding: '16px',
                        }}>
                          <Box sx={{ 
                            fontSize: '0.875rem', 
                            lineHeight: 1.5,
                            backgroundColor: 'white', 
                            p: 1.5, 
                            borderRadius: 1,
                            border: '1px solid',
                            borderColor: 'grey.100',
                            boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                          }}>
                            {objPreview.tokens.map((token, tokenIdx) => (
                              <HighlightedToken 
                                key={tokenIdx} 
                                token={token} 
                                tag={objPreview.tags[tokenIdx]} 
                              />
                            ))}
                          </Box>
                        </TableCell>
                        <TableCell>{objPreview.object}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              {/* Load more button for objects */}
              {hasMoreObjects && (
                <LoadMoreButton 
                  onClick={loadMoreObjects} 
                  isLoading={loadingMoreObjects} 
                />
              )}
            </>
          ) : (
            <Typography color="text.secondary">No objects found or still processing</Typography>
          )}
          
          {/* Show loading indicator if loading more objects */}
          {loadingMoreObjects && objectPreviews.length === 0 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress size={30} />
            </Box>
          )}
        </>
      ) : (
        // Classified Token view
        <>
          <Typography variant="h6" mb={2}>Sample Entities</Typography>
          
          {classifiedTokens.length > 0 ? (
            <>
              <TableContainer component={Paper} sx={{ maxHeight: 500 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Prediction</TableCell>
                      <TableCell>Source Object</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {classifiedTokens.map((entity, index) => (
                      <TableRow key={index}>
                        <TableCell 
                          sx={{ 
                            maxWidth: '70%',
                            minWidth: '400px'
                          }}
                        >
                          <Box sx={{ 
                            fontSize: '0.75rem',
                            fontFamily: 'monospace',
                            border: '1px solid',
                            borderColor: 'grey.200',
                            p: 1,
                            borderRadius: 1,
                            bgcolor: 'grey.50',
                            lineHeight: 1.5
                          }}>
                            <span style={{ color: 'rgb(102, 102, 102)' }}>
                              {entity.LContext || ''}
                            </span>
                            <HighlightedToken token={entity.Text} tag={entity.Label} />
                            <span style={{ color: 'rgb(102, 102, 102)' }}>
                              {entity.RContext || ''}
                            </span>
                          </Box>
                        </TableCell>
                        <TableCell>{entity.Object}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              {/* Load more button */}
              {hasMoreTokens && (
                <LoadMoreButton 
                  onClick={loadMoreTokens} 
                  isLoading={loadingMoreTokens} 
                />
              )}
            </>
          ) : (
            <Typography color="text.secondary">No entities found or still processing</Typography>
          )}
          
          {/* Show loading indicator if loading more tokens */}
          {loadingMoreTokens && classifiedTokens.length === 0 && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress size={30} />
            </Box>
          )}
        </>
      )}
    </Box>
  );
};

const JobDetail = ({ reportId, onBack }) => {
  const [report, setReport] = useState(null);
  const [entities, setEntities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('configuration');

  console.log('JobDetail render - reportId:', reportId);

  useEffect(() => {
    if (!reportId) {
      console.error('No reportId provided to JobDetail component');
      setError('No report ID provided');
      setLoading(false);
      return;
    }

    console.log('Fetching report details for reportId:', reportId);
    fetchReportDetails();
  }, [reportId]);

  const fetchReportDetails = async () => {
    try {
      setLoading(true);
      
      // Fetch report details
      console.log('Calling nerService.getReport with reportId:', reportId);
      const reportData = await nerService.getReport(reportId);
      console.log('Report data received:', reportData);
      setReport(reportData);
      
      // Fetch some example entities for this report
      try {
        console.log('Calling nerService.getReportEntities with reportId:', reportId);
        const entitiesData = await nerService.getReportEntities(reportId, { limit: 20 });
        console.log('Entities data received, count:', entitiesData.length);
        setEntities(entitiesData);
      } catch (entitiesError) {
        console.error('Error fetching entities:', entitiesError);
        // We don't set the main error here to still show the report details
      }
    } catch (err) {
      setError('Failed to fetch report details');
      console.error('Error fetching report details:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">Job Details</Typography>
            <Button variant="outlined" onClick={onBack}>Back to Jobs</Button>
          </Box>
          <Typography color="error">{error}</Typography>
        </CardContent>
      </Card>
    );
  }

  if (!report) {
    return (
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">Job Details</Typography>
            <Button variant="outlined" onClick={onBack}>Back to Jobs</Button>
          </Box>
          <Typography>No report found with ID: {reportId}</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2, alignItems: 'center' }}>
          <Typography variant="h6">{report.ReportName}</Typography>
          <Button variant="outlined" onClick={onBack}>Back to Jobs</Button>
        </Box>
        
        <Divider sx={{ mb: 2 }} />
        
        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange}
            aria-label="job detail tabs"
            sx={{
              '& .MuiTab-root': {
                textTransform: 'none',
                fontWeight: 500,
                fontSize: '1rem',
                color: '#5F6368',
                minWidth: 100,
                padding: '12px 16px',
                '&.Mui-selected': {
                  color: '#1a73e8',
                  fontWeight: 500
                }
              },
              '& .MuiTabs-indicator': {
                backgroundColor: '#1a73e8'
              }
            }}
          >
            <Tab label="Configuration" value="configuration" />
            <Tab label="Analytics" value="analytics" />
            <Tab label="Output" value="output" />
          </Tabs>
        </Box>
        
        {/* Tab Content */}
        <Box sx={{ py: 2 }}>
          {activeTab === 'configuration' && <ConfigurationTab report={report} />}
          {activeTab === 'analytics' && <AnalyticsTab report={report} />}
          {activeTab === 'output' && <OutputTab 
            entities={entities} 
            reportId={reportId} 
          />}
        </Box>
      </CardContent>
    </Card>
  );
};

export default JobDetail; 