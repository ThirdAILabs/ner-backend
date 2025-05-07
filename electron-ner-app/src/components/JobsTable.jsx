import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Card,
  CardContent,
} from '@mui/material';
import { format } from 'date-fns';
import { nerService } from '../lib/backend';

// Simplified for initial implementation
const ReportStatus = ({ report }) => {
  if (report.isLoadingStatus) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <CircularProgress size={16} />
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          Loading status...
        </Typography>
      </Box>
    );
  }

  if (!report.detailedStatus) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          Status unavailable
        </Typography>
      </Box>
    );
  }

  const { ShardDataTaskStatus, InferenceTaskStatuses } = report.detailedStatus;

  // Check for ShardDataTask failure first
  if (ShardDataTaskStatus === 'FAILED') {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <Box sx={{
          flex: 1,
          height: '8px',
          bgcolor: '#f1f5f9',
          borderRadius: '9999px',
          overflow: 'hidden'
        }}>
          <Box sx={{
            height: '100%',
            width: '100%',
            bgcolor: '#ef4444', // red color for failure
            borderRadius: '9999px',
          }} />
        </Box>
        <Typography variant="body2" sx={{ color: '#ef4444', whiteSpace: 'nowrap', fontWeight: 'medium' }}>
          Failed
        </Typography>
      </Box>
    );
  }

  // Calculate progress
  const completed = InferenceTaskStatuses?.COMPLETED?.TotalTasks || 0;
  const running = InferenceTaskStatuses?.RUNNING?.TotalTasks || 0;
  const queued = InferenceTaskStatuses?.QUEUED?.TotalTasks || 0;
  const failed = InferenceTaskStatuses?.FAILED?.TotalTasks || 0;

  const totalTasks = completed + running + queued + failed;
  const completedTasks = completed;
  const progress = totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0;

  // Show progress bar
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <Box sx={{
          flex: 1,
          height: '8px',
          bgcolor: '#f1f5f9',
          borderRadius: '9999px',
          overflow: 'hidden'
        }}>
          <Box sx={{
            height: '100%',
            width: `${progress}%`,
            bgcolor: ShardDataTaskStatus === 'COMPLETED' && progress === 100 ? '#4caf50' : '#1976d2',
            borderRadius: '9999px',
            transition: 'all 0.2s'
          }} />
        </Box>
        <Typography variant="body2" sx={{ color: 'text.secondary', whiteSpace: 'nowrap' }}>
          {`${Math.round(progress)}%`}
        </Typography>
      </Box>
      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
        {`${completedTasks} completed, ${running} running, ${queued} queued`}
      </Typography>
    </Box>
  );
};

const JobsTable = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchReportStatus = async (report) => {
    try {
      // Mark report as loading status
      setReports(prev =>
        prev.map(r =>
          r.Id === report.Id
            ? { ...r, isLoadingStatus: true }
            : r
        )
      );

      // Fetch detailed report
      const detailedReport = await nerService.getReport(report.Id);

      // Update report with detailed status
      setReports(prev =>
        prev.map(r =>
          r.Id === report.Id
            ? {
              ...r,
              detailedStatus: {
                ShardDataTaskStatus: detailedReport.ShardDataTaskStatus,
                InferenceTaskStatuses: detailedReport.InferenceTaskStatuses
              },
              isLoadingStatus: false
            }
            : r
        )
      );
    } catch (err) {
      console.error(`Error fetching status for report ${report.Id}:`, err);
      
      // Mark report as not loading even if there was an error
      setReports(prev =>
        prev.map(r =>
          r.Id === report.Id
            ? { ...r, isLoadingStatus: false }
            : r
        )
      );
    }
  };

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        
        // Fetch all reports
        const reportsData = await nerService.listReports();
        setReports(reportsData);

        // Fetch detailed status for each report
        reportsData.forEach(report => {
          fetchReportStatus(report);
        });
      } catch (err) {
        setError('Failed to fetch reports');
        console.error('Error fetching reports:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchReports();
  }, []);

  // Loading state
  if (loading) {
    return (
      <Card sx={{ boxShadow: '0 1px 3px rgba(0,0,0,0.1)', bgcolor: 'white' }}>
        <CardContent sx={{ p: 3 }}>
          <Box sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 3
          }}>
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.125rem' }}>
              Jobs
            </Typography>
            <Button
              variant="contained"
              color="primary"
              sx={{
                bgcolor: '#1976d2',
                '&:hover': {
                  bgcolor: '#1565c0',
                },
                textTransform: 'none',
                fontWeight: 500
              }}
            >
              New
            </Button>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={24} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (error) {
    return (
      <Card sx={{ boxShadow: '0 1px 3px rgba(0,0,0,0.1)', bgcolor: 'white' }}>
        <CardContent sx={{ p: 3 }}>
          <Box sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 3
          }}>
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.125rem' }}>
              Jobs
            </Typography>
            <Button
              variant="contained"
              color="primary"
              sx={{
                bgcolor: '#1976d2',
                '&:hover': {
                  bgcolor: '#1565c0',
                },
                textTransform: 'none',
                fontWeight: 500
              }}
            >
              New
            </Button>
          </Box>
          <Typography sx={{ textAlign: 'center', py: 2, color: 'error.main' }}>
            {error}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  // Main table view
  return (
    <Card sx={{ boxShadow: '0 1px 3px rgba(0,0,0,0.1)', bgcolor: 'white' }}>
      <CardContent sx={{ p: 3 }}>
        <Box sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3
        }}>
          <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.125rem' }}>
            Jobs
          </Typography>
          <Button
            variant="contained"
            color="primary"
            sx={{
              bgcolor: '#1976d2',
              '&:hover': {
                bgcolor: '#1565c0',
              },
              textTransform: 'none',
              fontWeight: 500
            }}
          >
            New
          </Button>
        </Box>

        <TableContainer
          component={Paper}
          sx={{
            boxShadow: 'none',
            border: '1px solid rgba(0, 0, 0, 0.12)',
            borderRadius: '0.375rem',
            overflow: 'hidden'
          }}
        >
          <Table>
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 600 }}>Name</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Model</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Created At</TableCell>
                <TableCell sx={{ fontWeight: 600 }}>Action</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {reports && reports.length > 0 ? (
                reports.map((report, index) => (
                  <TableRow
                    key={report.Id}
                    sx={{
                      bgcolor: index % 2 === 0 ? 'white' : '#f9fafb'
                    }}
                  >
                    <TableCell>{report.ReportName}</TableCell>
                    <TableCell>{report.Model.Name}</TableCell>
                    <TableCell>
                      <ReportStatus report={report} />
                    </TableCell>
                    <TableCell>{format(new Date(report.CreationTime), 'MM/dd/yyyy, hh:mm:ss a')}</TableCell>
                    <TableCell>
                      <Typography
                        onClick={() => console.log(`View report: ${report.Id}`)}
                        sx={{
                          color: '#1976d2',
                          cursor: 'pointer',
                          '&:hover': {
                            textDecoration: 'underline'
                          }
                        }}
                      >
                        View
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={5}
                    sx={{
                      textAlign: 'center',
                      py: 2,
                      color: 'text.secondary'
                    }}
                  >
                    No reports found. Create a new report to get started.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default JobsTable; 