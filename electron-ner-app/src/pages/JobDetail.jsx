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
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';

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