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
} from '@mui/material';
import { format } from 'date-fns';
import { nerService } from '../lib/backend';

// Configuration tab component
const ConfigurationTab = ({ report }) => {
  if (!report) return null;
  
  return (
    <Box sx={{ mt: 3 }}>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" fontWeight="bold">Name</Typography>
          <Typography mb={2}>{report.ReportName}</Typography>
          
          <Typography variant="subtitle1" fontWeight="bold">Model</Typography>
          <Typography mb={2}>{report.Model?.Name}</Typography>
          
          <Typography variant="subtitle1" fontWeight="bold">Created At</Typography>
          <Typography mb={2}>{format(new Date(report.CreationTime), 'PPpp')}</Typography>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography variant="subtitle1" fontWeight="bold">Source</Typography>
          <Typography mb={2}>{report.SourceS3Bucket}/{report.SourceS3Prefix || 'N/A'}</Typography>
          
          <Typography variant="subtitle1" fontWeight="bold">Tags</Typography>
          <Box sx={{ mb: 2 }}>
            {report.Tags && report.Tags.length > 0 ? (
              report.Tags.map((tag, index) => (
                <Chip key={index} label={tag} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
              ))
            ) : (
              <Typography color="text.secondary">No tags</Typography>
            )}
          </Box>
        </Grid>
      </Grid>
      
      {report.Groups && report.Groups.length > 0 && (
        <>
          <Typography variant="subtitle1" fontWeight="bold" mt={3} mb={1}>Groups</Typography>
          <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Query</TableCell>
                  <TableCell>Count</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {report.Groups.map((group, index) => (
                  <TableRow key={index}>
                    <TableCell>{group.Name}</TableCell>
                    <TableCell>{group.Query}</TableCell>
                    <TableCell>{group.Objects?.length || 0}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
    </Box>
  );
};

// Analytics tab component
const AnalyticsTab = ({ report }) => {
  if (!report) return null;
  
  // Calculate completion status
  const getStatusDisplay = () => {
    if (!report.InferenceTaskStatuses) {
      return 'Pending';
    }
    
    const completed = report.InferenceTaskStatuses?.COMPLETED?.TotalTasks || 0;
    const running = report.InferenceTaskStatuses?.RUNNING?.TotalTasks || 0;
    const queued = report.InferenceTaskStatuses?.QUEUED?.TotalTasks || 0;
    const failed = report.InferenceTaskStatuses?.FAILED?.TotalTasks || 0;
    
    const totalTasks = completed + running + queued + failed;
    
    if (failed > 0) {
      return 'Failed';
    }
    
    if (totalTasks === 0) {
      return report.ShardDataTaskStatus || 'Pending';
    }
    
    if (completed === totalTasks) {
      return 'Completed';
    }
    
    return `In Progress (${Math.round((completed / totalTasks) * 100)}%)`;
  };
  
  const getProgressPercentage = () => {
    if (!report.InferenceTaskStatuses) return 0;
    
    const completed = report.InferenceTaskStatuses?.COMPLETED?.TotalTasks || 0;
    const running = report.InferenceTaskStatuses?.RUNNING?.TotalTasks || 0;
    const queued = report.InferenceTaskStatuses?.QUEUED?.TotalTasks || 0;
    const failed = report.InferenceTaskStatuses?.FAILED?.TotalTasks || 0;
    
    const totalTasks = completed + running + queued + failed;
    return totalTasks > 0 ? Math.round((completed / totalTasks) * 100) : 0;
  };
  
  return (
    <Box sx={{ mt: 3 }}>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 2, bgcolor: '#f5f5f5' }}>
            <Typography variant="subtitle1" fontWeight="bold">Status</Typography>
            <Box sx={{ mt: 2, mb: 1 }}>
              <Chip 
                label={getStatusDisplay()} 
                color={
                  getStatusDisplay() === 'Failed' ? 'error' :
                  getStatusDisplay() === 'Completed' ? 'success' :
                  'primary'
                }
              />
            </Box>
            
            {/* Progress bar */}
            {getStatusDisplay().includes('In Progress') && (
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Typography variant="body2" fontWeight="medium">Progress</Typography>
                  <Typography variant="body2" ml="auto">{getProgressPercentage()}%</Typography>
                </Box>
                <Box sx={{ 
                  width: '100%', 
                  height: 8, 
                  bgcolor: '#e0e0e0', 
                  borderRadius: 1, 
                  overflow: 'hidden' 
                }}>
                  <Box sx={{ 
                    width: `${getProgressPercentage()}%`, 
                    height: '100%', 
                    bgcolor: '#1976d2',
                    borderRadius: 1
                  }} />
                </Box>
              </Box>
            )}
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ p: 2, bgcolor: '#f5f5f5' }}>
            <Typography variant="subtitle1" fontWeight="bold">Entity Counts</Typography>
            {report.TagCounts && Object.keys(report.TagCounts).length > 0 ? (
              <Box sx={{ mt: 2 }}>
                {Object.entries(report.TagCounts).map(([tag, count], index) => (
                  <Chip 
                    key={index} 
                    label={`${tag}: ${count}`} 
                    size="small" 
                    sx={{ mr: 0.5, mb: 0.5 }} 
                  />
                ))}
              </Box>
            ) : (
              <Typography color="text.secondary" mt={1}>No entity counts available</Typography>
            )}
          </Card>
        </Grid>
      </Grid>
      
      {/* Task Statistics */}
      <Card sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5' }}>
        <Typography variant="subtitle1" fontWeight="bold">Task Statistics</Typography>
        <Grid container spacing={2} mt={0.5}>
          <Grid item xs={6} md={3}>
            <Typography variant="body2" color="text.secondary">Completed Tasks</Typography>
            <Typography variant="h6">{report.InferenceTaskStatuses?.COMPLETED?.TotalTasks || 0}</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="body2" color="text.secondary">Running Tasks</Typography>
            <Typography variant="h6">{report.InferenceTaskStatuses?.RUNNING?.TotalTasks || 0}</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="body2" color="text.secondary">Queued Tasks</Typography>
            <Typography variant="h6">{report.InferenceTaskStatuses?.QUEUED?.TotalTasks || 0}</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="body2" color="text.secondary">Failed Tasks</Typography>
            <Typography variant="h6" color={report.InferenceTaskStatuses?.FAILED?.TotalTasks > 0 ? 'error' : 'inherit'}>
              {report.InferenceTaskStatuses?.FAILED?.TotalTasks || 0}
            </Typography>
          </Grid>
        </Grid>
      </Card>
    </Box>
  );
};

// Output tab component
const OutputTab = ({ entities }) => {
  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" mb={2}>Sample Entities</Typography>
      
      {entities.length > 0 ? (
        <TableContainer component={Paper} sx={{ maxHeight: 500 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell>Object</TableCell>
                <TableCell>Text</TableCell>
                <TableCell>Label</TableCell>
                <TableCell>Position</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {entities.map((entity, index) => (
                <TableRow key={index}>
                  <TableCell>{entity.Object}</TableCell>
                  <TableCell>
                    {entity.LContext && <span style={{ color: '#999' }}>{entity.LContext}</span>}
                    <span style={{ fontWeight: 'bold', color: '#3f51b5' }}>{entity.Text}</span>
                    {entity.RContext && <span style={{ color: '#999' }}>{entity.RContext}</span>}
                  </TableCell>
                  <TableCell>
                    <Chip label={entity.Label} size="small" />
                  </TableCell>
                  <TableCell>{entity.Start}-{entity.End}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      ) : (
        <Typography color="text.secondary">No entities found or still processing</Typography>
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

  useEffect(() => {
    const fetchReportDetails = async () => {
      try {
        setLoading(true);
        
        // Fetch report details
        const reportData = await nerService.getReport(reportId);
        setReport(reportData);
        
        // Fetch some example entities for this report
        try {
          const entitiesData = await nerService.getReportEntities(reportId, { limit: 20 });
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

    fetchReportDetails();
  }, [reportId]);

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
          {activeTab === 'output' && <OutputTab entities={entities} />}
        </Box>
      </CardContent>
    </Card>
  );
};

export default JobDetail; 