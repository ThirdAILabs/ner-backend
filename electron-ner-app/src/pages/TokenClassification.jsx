import React, { useState, useEffect } from 'react';
import { Box, Typography, Tabs, Tab, CircularProgress } from '@mui/material';
import JobsTable from '../components/JobsTable';
import JobDetail from './JobDetail';
import CreateJob from './CreateJob';

// These components would be imported from their actual locations
// For now we'll create placeholder components
const Dashboard = () => (
  <Box sx={{ padding: '24px', backgroundColor: '#F5F7FA', minHeight: 'calc(100vh - 112px)' }}>
    <div className="space-y-6">
      <Box sx={{ p: 2, bgcolor: 'white', borderRadius: 1, boxShadow: 1 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Training Results</Typography>
        <Typography>Performance metrics and training outcomes would be displayed here.</Typography>
      </Box>
      
      <Box sx={{ p: 2, bgcolor: 'white', borderRadius: 1, boxShadow: 1 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Examples Visualizer</Typography>
        <Typography>Visualization of example classifications would appear here.</Typography>
      </Box>
      
      <Box sx={{ p: 2, bgcolor: 'white', borderRadius: 1, boxShadow: 1 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>Model Update</Typography>
        <Typography>Options to update the model would be available here.</Typography>
      </Box>
    </div>
  </Box>
);

const TokenClassification = () => {
  const [tabValue, setTabValue] = useState('monitoring');
  const [isLoading, setIsLoading] = useState(true);
  const [selectedReportId, setSelectedReportId] = useState(null);
  const [showCreateJob, setShowCreateJob] = useState(false);
  const workflowName = 'PII'; // Default workflow name

  useEffect(() => {
    // Simulate loading for a short time
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);
    
    return () => clearTimeout(timer);
  }, []);

  const handleTabChange = (_event, newValue) => {
    setTabValue(newValue);
  };

  const handleViewReport = (reportId) => {
    setSelectedReportId(reportId);
    setShowCreateJob(false);
  };

  const handleBackToJobs = () => {
    setSelectedReportId(null);
    setShowCreateJob(false);
  };
  
  const handleCreateNewJob = () => {
    setShowCreateJob(true);
    setSelectedReportId(null);
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <div style={{ backgroundColor: '#F5F7FA', minHeight: '100vh' }}>
      <header style={{ width: '100%', padding: '16px', borderBottom: '1px solid #e0e0e0' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', marginBottom: '16px' }}>
          <Typography
            variant="body1"
            color="textSecondary"
            gutterBottom
            style={{ fontWeight: 500 }}
          >
            Token Classification
          </Typography>

          <Typography
            variant="h5"
            style={{
              fontWeight: 'bold',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}
            title={workflowName}
          >
            {workflowName}
          </Typography>
        </div>
      </header>

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '16px' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            aria-label="token classification tabs"
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
            <Tab label="Monitoring" value="monitoring" />
            <Tab label="Jobs" value="jobs" />
          </Tabs>
        </Box>

        {/* Tab Content Sections */}
        <div style={{ display: tabValue === 'monitoring' ? 'block' : 'none' }}>
          <Dashboard />
        </div>

        <div style={{ display: tabValue === 'jobs' ? 'block' : 'none' }}>
          {showCreateJob ? (
            <CreateJob onBack={handleBackToJobs} />
          ) : selectedReportId ? (
            <JobDetail reportId={selectedReportId} onBack={handleBackToJobs} />
          ) : (
            <JobsTable onViewReport={handleViewReport} onCreateNewJob={handleCreateNewJob} />
          )}
        </div>
      </main>
    </div>
  );
};

export default TokenClassification; 