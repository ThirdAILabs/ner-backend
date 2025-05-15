'use client';

import { useEffect, useState } from 'react';
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
  CircularProgress
} from '@mui/material';
import { Plus } from 'lucide-react';

import { Card, CardContent } from '@mui/material';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { format } from 'date-fns';
import { nerService } from '@/lib/backend';
import { IconButton, Tooltip } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { useHealth } from '@/contexts/HealthProvider';
// Import the TaskStatusCategory from backend.ts
interface TaskStatusCategory {
  TotalTasks: number;
  TotalSize: number;
}

// Using the Report interface from backend.ts
interface ReportWithStatus {
  Id: string;
  Model: {
    Id: string;
    Name: string;
    Status: string;
    BaseModelId?: string;
    Tags?: string[];
  };
  SourceS3Bucket: string;
  SourceS3Prefix?: string;
  CreationTime: string;
  FileCount: number;
  CompletedFileCount: number;
  Tags?: string[];
  CustomTags?: { [key: string]: string };
  Groups?: {
    Id: string;
    Name: string;
    Query: string;
    Objects?: string[];
  }[];
  ShardDataTaskStatus?: string;
  InferenceTaskStatuses?: { [key: string]: TaskStatusCategory };
  Errors?: string[];

  // Additional fields for UI
  detailedStatus?: {
    ShardDataTaskStatus?: string;
    InferenceTaskStatuses?: { [key: string]: TaskStatusCategory };
  };
  isLoadingStatus?: boolean;
  ReportName: string;
}

export default function Jobs() {
  const searchParams = useSearchParams();
  const deploymentId = searchParams.get('deploymentId');
  const [reports, setReports] = useState<ReportWithStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { healthStatus } = useHealth();

  const fetchReportStatus = async (report: ReportWithStatus) => {
    try {
      setReports((prev) =>
        prev.map((r) =>
          r.Id === report.Id ? { ...r, isLoadingStatus: true } : r
        )
      );

      const detailedReport = await nerService.getReport(report.Id);

      setReports((prev) =>
        prev.map((r) =>
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
      setReports((prev) =>
        prev.map((r) =>
          r.Id === report.Id ? { ...r, isLoadingStatus: false } : r
        )
      );
    }
  };


  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        const reportsData = await nerService.listReports();
        reportsData.sort(
          (a: ReportWithStatus, b: ReportWithStatus) =>
            new Date(b.CreationTime).getTime() -
            new Date(a.CreationTime).getTime()
        );
        setReports(reportsData as ReportWithStatus[]);

        // Fetch status for each report
        reportsData.forEach((report) => {
          fetchReportStatus(report as ReportWithStatus);
        });
      } catch (err) {
        setError('Failed to fetch reports');
        console.error('Error fetching reports:', err);
      } finally {
        setLoading(false);
      }
    };
    if (healthStatus)
      fetchReports();
  }, [healthStatus]);

  if (loading) {
    return (
      <Card sx={{ boxShadow: '0 1px 3px rgba(0,0,0,0.1)', bgcolor: 'white' }}>
        <CardContent sx={{ p: 3 }}>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 3
            }}
          >
            <Typography
              variant="h6"
              sx={{ fontWeight: 600, fontSize: '1.125rem' }}
            >
              Report
            </Typography>
            <Link href={`/token-classification/jobs/new`} passHref>
              <Button
                variant="contained"
                color="primary"
                sx={{
                  bgcolor: '#1976d2',
                  '&:hover': {
                    bgcolor: '#1565c0'
                  },
                  textTransform: 'none',
                  fontWeight: 500
                }}
              >
                <Plus size={16} />
              </Button>
            </Link>
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={24} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ boxShadow: '0 1px 3px rgba(0,0,0,0.1)', bgcolor: 'white' }}>
        <CardContent sx={{ p: 3 }}>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 3
            }}
          >
            <Typography
              variant="h6"
              sx={{ fontWeight: 600, fontSize: '1.125rem' }}
            >
              Report
            </Typography>
            <Link href={`/token-classification/jobs/new`} passHref>
              <Button
                variant="contained"
                color="primary"
                sx={{
                  bgcolor: '#1976d2',
                  '&:hover': {
                    bgcolor: '#1565c0'
                  },
                  textTransform: 'none',
                  fontWeight: 500
                }}
              >
                <Plus size={16} />
              </Button>
            </Link>
          </Box>
          <Typography sx={{ textAlign: 'center', py: 2, color: 'error.main' }}>
            {error}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const getStatusDisplay = (report: ReportWithStatus) => {
    if (report.isLoadingStatus) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <CircularProgress size={16} />
          {/* <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Loading status...
          </Typography> */}
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

    const { ShardDataTaskStatus, InferenceTaskStatuses } =
      report.detailedStatus;

    // Check for ShardDataTask failure first
    if (ShardDataTaskStatus === 'FAILED') {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box
            sx={{
              flex: 1,
              height: '8px',
              bgcolor: '#f1f5f9',
              borderRadius: '9999px',
              overflow: 'hidden'
            }}
          >
            <Box
              sx={{
                height: '100%',
                width: '100%',
                bgcolor: '#ef4444', // red color for failure
                borderRadius: '9999px'
              }}
            />
          </Box>
          <Typography
            variant="body2"
            sx={{
              color: '#ef4444',
              whiteSpace: 'nowrap',
              fontWeight: 'medium'
            }}
          >
            Failed (Data Sharding Error)
          </Typography>
        </Box>
      );
    }

    // Add default values for each status
    const completed = InferenceTaskStatuses?.COMPLETED?.TotalTasks || 0;
    const running = InferenceTaskStatuses?.RUNNING?.TotalTasks || 0;
    const queued = InferenceTaskStatuses?.QUEUED?.TotalTasks || 0;
    const failed = InferenceTaskStatuses?.FAILED?.TotalTasks || 0;

    const fileCount = report.FileCount || 0;
    const completedFileCount = report.CompletedFileCount || 0;
    const totalTasks = completed + running + queued + failed;

    const progress = completedFileCount > 0 ? (completedFileCount / fileCount) * 100 : 0;

    // If there are failed tasks, show failure status
    if (failed > 0) {
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Box
              sx={{
                flex: 1,
                height: '8px',
                bgcolor: '#f1f5f9',
                borderRadius: '9999px',
                overflow: 'hidden'
              }}
            >
              <Box
                sx={{
                  height: '100%',
                  width: '100%',
                  bgcolor: '#ef4444',
                  borderRadius: '9999px'
                }}
              />
            </Box>
            <Typography
              variant="body2"
              sx={{
                color: '#ef4444',
                whiteSpace: 'nowrap',
                fontWeight: 'medium'
              }}
            >
              Failed
            </Typography>
          </Box>
          <Typography variant="caption" sx={{ color: '#ef4444' }}>
            {`${failed} task${failed > 1 ? 's' : ''} failed out of ${totalTasks} total`}
          </Typography>
        </Box>
      );
    }

    // If no tasks yet, show just the ShardDataTaskStatus
    if (totalTasks === 0) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            {ShardDataTaskStatus || 'PENDING'}
          </Typography>
        </Box>
      );
    }

    // Show progress for running tasks
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box
            sx={{
              flex: 1,
              height: '8px',
              bgcolor: '#f1f5f9',
              borderRadius: '9999px',
              overflow: 'hidden'
            }}
          >
            <Box
              sx={{
                height: '100%',
                width: `${progress}%`,
                bgcolor:
                  ShardDataTaskStatus === 'COMPLETED' && progress === 100
                    ? '#4caf50'
                    : '#1976d2',
                borderRadius: '9999px',
                transition: 'all 0.2s'
              }}
            />
          </Box>
          <Typography
            variant="body2"
            sx={{ color: 'text.secondary', whiteSpace: 'nowrap' }}
          >
            {`${Math.round(progress)}%`}
          </Typography>
        </Box>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          {`Files: ${completedFileCount}/${fileCount}`}
        </Typography>
      </Box>
    );
  };

  const handleDelete = async (reportId: string) => {
    if (window.confirm('Are you sure you want to delete this report?')) {
      try {
        await nerService.deleteReport(reportId);
        // Update the reports list after deletion
        setReports(reports.filter((report) => report.Id !== reportId));
      } catch (error) {
        console.error('Error deleting report:', error);
        // You might want to show an error message to the user
      }
    }
  };

  return (
    <Card sx={{ boxShadow: '0 1px 3px rgba(0,0,0,0.1)', bgcolor: 'grey.100' }}>
      <CardContent sx={{ p: 3 }}>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 3,
          }}
        >
          <Typography
            variant="h6"
            sx={{ fontWeight: 500, fontSize: '1.25rem' }}
          >
            Reports
          </Typography>
          <Link href={`/token-classification/jobs/new`} passHref>
            <Button
              variant="contained"
              color="primary"
              sx={{
                bgcolor: '#1976d2',
                '&:hover': {
                  bgcolor: '#1565c0'
                },
                textTransform: 'none',
                fontWeight: 500,
                width: 48,
                height: 48,
                minWidth: 0,
                padding: 0,
                borderRadius: '50%'
              }}
              disabled={!healthStatus}
            >
              <Plus size={24} />
            </Button>
          </Link>
        </Box>

        <TableContainer
          component={Paper}
          sx={{
            boxShadow: 'none',
            border: '1px solid rgba(0, 0, 0, 0.12)',
            borderRadius: '0.375rem',
            overflow: 'hidden',
            // width: '80%',
            // margin: '0 auto',
          }}
        >
          <Table>
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 600, width: '20%' }}>
                  Name
                </TableCell>
                <TableCell sx={{ fontWeight: 600, width: '15%' }}>
                  Model
                </TableCell>
                <TableCell sx={{ fontWeight: 600, width: '35%' }}>
                  Progress
                </TableCell>
                <TableCell sx={{ fontWeight: 600, width: '20%' }}>
                  Created At
                </TableCell>
                <TableCell sx={{ fontWeight: 600, width: '10%' }}>
                  Actions
                </TableCell>
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
                    <TableCell>{getStatusDisplay(report)}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                        <Typography variant="body2">
                          {format(
                            new Date(report.CreationTime),
                            'MMMM d, yyyy'
                          )}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {format(new Date(report.CreationTime), 'h:mm a')}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box
                        sx={{ display: 'flex', alignItems: 'center', gap: 2 }}
                      >
                        <Link
                          href={`/token-classification/jobs?jobId=${report.Id}`}
                          style={{
                            color: '#1976d2',
                            textDecoration: 'none'
                          }}
                        >
                          <Typography
                            sx={{
                              '&:hover': {
                                textDecoration: 'underline'
                              }
                            }}
                          >
                            View
                          </Typography>
                        </Link>
                        <Tooltip title="Delete report">
                          <IconButton
                            size="small"
                            onClick={() => handleDelete(report.Id)}
                            sx={{
                              color: '#dc2626',
                              '&:hover': {
                                bgcolor: 'rgba(220, 38, 38, 0.04)'
                              }
                            }}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
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
}
