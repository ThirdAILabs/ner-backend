'use client';

import { useEffect, useRef, useState } from 'react';
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
import { alpha } from '@mui/material/styles';

import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

export default function Jobs() {
  const searchParams = useSearchParams();
  const deploymentId = searchParams.get('deploymentId');
  const [reports, setReports] = useState<ReportWithStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { healthStatus } = useHealth();
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchReportStatus = async (report: ReportWithStatus) => {
    const pollStatus = async () => {
      try {
        setReports((prev) =>
          prev.map((r) => (r.Id === report.Id ? { ...r, isLoadingStatus: true } : r))
        );

        const detailedReport = await nerService.getReport(report.Id);

        setReports((prev) =>
          prev.map((r) =>
            r.Id === report.Id
              ? {
                  ...r,
                  Errors: detailedReport.Errors || [],
                  SucceededFileCount: detailedReport.SucceededFileCount,
                  FailedFileCount: detailedReport.FailedFileCount,
                  detailedStatus: {
                    ShardDataTaskStatus: detailedReport.ShardDataTaskStatus,
                    InferenceTaskStatuses: detailedReport.InferenceTaskStatuses,
                  },
                  isLoadingStatus: false,
                }
              : r
          )
        );

        // If files are complete, return true to stop polling
        return (
          detailedReport.SucceededFileCount + detailedReport.FailedFileCount ===
          detailedReport.FileCount
        );
      } catch (err) {
        console.error(`Error fetching status for report ${report.Id}:`, err);
        setReports((prev) =>
          prev.map((r) => (r.Id === report.Id ? { ...r, isLoadingStatus: false } : r))
        );
        return false;
      }
    };

    const poll = async () => {
      const isComplete = await pollStatus();

      if (isComplete) {
        clearInterval(pollIntervalRef.current!);
      }
    };

    poll();
    pollIntervalRef.current = setInterval(poll, 5000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  };

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        const reportsData = await nerService.listReports();
        reportsData.sort(
          (a: ReportWithStatus, b: ReportWithStatus) =>
            new Date(b.CreationTime).getTime() - new Date(a.CreationTime).getTime()
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
    if (healthStatus) fetchReports();
    return () => {};
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
              mb: 3,
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.125rem' }}>
              Scan
            </Typography>
            <Link href={`/token-classification/jobs/new`} passHref>
              <Button
                variant="contained"
                color="primary"
                sx={{
                  bgcolor: '#1976d2',
                  '&:hover': {
                    bgcolor: '#1565c0',
                  },
                  textTransform: 'none',
                  fontWeight: 500,
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
              mb: 3,
            }}
          >
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.125rem' }}>
              Scan
            </Typography>
            <Link href={`/token-classification/jobs/new`} passHref>
              <Button
                variant="contained"
                color="primary"
                sx={{
                  bgcolor: '#1976d2',
                  '&:hover': {
                    bgcolor: '#1565c0',
                  },
                  textTransform: 'none',
                  fontWeight: 500,
                }}
              >
                <Plus size={16} />
              </Button>
            </Link>
          </Box>
          <Typography sx={{ textAlign: 'center', py: 2, color: 'error.main' }}>{error}</Typography>
        </CardContent>
      </Card>
    );
  }

  const getStatusDisplay = (report: ReportWithStatus) => {
    if (report.isLoadingStatus) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <CircularProgress size={16} />
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              Loading status...
            </Typography>
          </Typography>
        </Box>
      );
    }

    if (!report.detailedStatus) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <CircularProgress size={16} />
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            Loading status...
          </Typography>
        </Box>
      );
    }

    const { ShardDataTaskStatus, InferenceTaskStatuses } = report.detailedStatus;
    const reportErrors = report.Errors || [];
    const monthlyQuotaExceeded =
      reportErrors.length > 0 &&
      reportErrors.includes('license verification failed: quota exceeded');

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
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                height: '100%',
                width: '100%',
                bgcolor: '#ef4444', // red color for failure
                borderRadius: '9999px',
              }}
            />
          </Box>
          <Typography
            variant="body2"
            sx={{
              color: '#ef4444',
              whiteSpace: 'nowrap',
              fontWeight: 'medium',
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
    const succeededFileCount = report.SucceededFileCount || 0;
    const failedFileCount = report.FailedFileCount || 0;
    const totalTasks = completed + running + queued + failed;

    const progress = succeededFileCount > 0 ? (succeededFileCount / fileCount) * 100 : 0;

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

    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box
            sx={{
              flex: 1,
              height: '8px',
              bgcolor: '#cbd5e1',
              borderRadius: '9999px',
              overflow: 'hidden',
              display: 'flex',
              position: 'relative',
            }}
          >
            {/* Green (successful files) */}
            <Box
              sx={{
                height: '100%',
                width: `${(succeededFileCount / fileCount) * 100}%`,
                bgcolor: '#4caf50',
              }}
            />
            {/* Red (failed files) */}
            <Box
              sx={{
                height: '100%',
                width: `${(failedFileCount / fileCount) * 100}%`,
                bgcolor: '#ef4444',
              }}
            />
            {/* Loading animation */}
            {!monthlyQuotaExceeded && succeededFileCount + failedFileCount < fileCount && (
              <Box
                className="shimmer-effect"
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                }}
              />
            )}
          </Box>
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary',
              whiteSpace: 'nowrap',
              fontWeight: 'medium',
            }}
          >
            {`${(((succeededFileCount + failedFileCount) / fileCount) * 100).toFixed(0)} %`}
          </Typography>
        </Box>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          {monthlyQuotaExceeded ? (
            <Box
              component="span"
              sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'red' }}
            >
              Monthly Quota Exceeded
              <Tooltip title="This scan exceeds the monthly quota for the free-tier. The quota resets on the 1st of each month.">
                <InfoOutlinedIcon fontSize="inherit" sx={{ cursor: 'pointer' }} />
              </Tooltip>
            </Box>
          ) : succeededFileCount === fileCount ? (
            `Files: ${fileCount}/${fileCount} Processed`
          ) : failedFileCount === fileCount ? (
            `Files: ${failedFileCount}/${fileCount} Failed`
          ) : (
            `Files: ${succeededFileCount}/${fileCount} Processed${failedFileCount > 0 ? `, ${failedFileCount}/${fileCount} Failed` : ''}`
          )}
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
    <Card
      sx={{
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        bgcolor: 'white',
        borderRadius: '12px',
        mx: 'auto',
        maxWidth: '1400px',
      }}
    >
      <CardContent sx={{ p: 4 }}>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 4,
          }}
        >
          <Box>
            <Typography
              variant="h5"
              sx={{
                fontWeight: 600,
                fontSize: '1.5rem',
                color: '#4a5568',
              }}
            >
              Scans
            </Typography>
          </Box>
          <Link href={`/token-classification/jobs/new`} passHref>
            <Button
              variant="contained"
              color="primary"
              startIcon={<Plus size={20} />}
              sx={{
                bgcolor: '#2563eb',
                '&:hover': {
                  bgcolor: '#1d4ed8',
                },
                textTransform: 'none',
                fontWeight: 500,
                px: 3,
                py: 1.5,
                borderRadius: '8px',
                boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
              }}
              disabled={!healthStatus}
            >
              New Scan
            </Button>
          </Link>
        </Box>

        <TableContainer
          component={Paper}
          sx={{
            boxShadow: 'none',
            border: '1px solid',
            borderColor: 'grey.200',
            borderRadius: '12px',
            bgcolor: 'white',
            '& .MuiTableCell-root': {
              borderBottom: '1px solid',
              borderColor: 'grey.200',
              py: 2.5,
            },
          }}
        >
          <Table>
            <TableHead>
              <TableRow sx={{ bgcolor: '#f8fafc' }}>
                <TableCell
                  sx={{
                    fontWeight: 600,
                    color: '#475569',
                    fontSize: '0.875rem',
                  }}
                >
                  Name
                </TableCell>
                <TableCell
                  sx={{
                    fontWeight: 600,
                    color: '#475569',
                    fontSize: '0.875rem',
                  }}
                >
                  Model
                </TableCell>
                <TableCell
                  sx={{
                    fontWeight: 600,
                    color: '#475569',
                    fontSize: '0.875rem',
                  }}
                >
                  Progress
                </TableCell>
                <TableCell
                  sx={{
                    fontWeight: 600,
                    color: '#475569',
                    fontSize: '0.875rem',
                  }}
                >
                  Created At
                </TableCell>
                <TableCell
                  sx={{
                    fontWeight: 600,
                    color: '#475569',
                    fontSize: '0.875rem',
                    width: '120px',
                  }}
                >
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
                      bgcolor: 'white',
                      '&:hover': {
                        bgcolor: alpha('#60a5fa', 0.04),
                      },
                      transition: 'background-color 0.2s',
                    }}
                  >
                    <TableCell>
                      <Typography
                        sx={{
                          fontWeight: 500,
                          color: '#1e293b',
                        }}
                      >
                        {report.ReportName}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box
                        sx={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          px: 2,
                          py: 0.5,
                          bgcolor: '#f1f5f9',
                          borderRadius: '16px',
                        }}
                      >
                        <Typography
                          sx={{
                            fontSize: '0.875rem',
                            color: '#475569',
                          }}
                        >
                          {report.Model.Name.charAt(0).toUpperCase() + report.Model.Name.slice(1)}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>{getStatusDisplay(report)}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                        <Typography variant="body2" sx={{ color: '#1e293b' }}>
                          {format(new Date(report.CreationTime), 'MMMM d, yyyy')}
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#64748b' }}>
                          {format(new Date(report.CreationTime), 'h:mm a')}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 1,
                        }}
                      >
                        <Link
                          href={`/token-classification/jobs?jobId=${report.Id}`}
                          style={{
                            textDecoration: 'none',
                          }}
                        >
                          <Button
                            variant="outlined"
                            size="small"
                            sx={{
                              borderColor: '#e2e8f0',
                              color: '#475569',
                              '&:hover': {
                                borderColor: '#cbd5e1',
                                bgcolor: '#f8fafc',
                              },
                              textTransform: 'none',
                              minWidth: 0,
                              px: 2,
                            }}
                          >
                            View
                          </Button>
                        </Link>
                        <Tooltip title="Delete report">
                          <IconButton
                            size="small"
                            onClick={() => handleDelete(report.Id)}
                            sx={{
                              color: '#dc2626',
                              '&:hover': {
                                bgcolor: alpha('#dc2626', 0.04),
                              },
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
                      py: 8,
                      textAlign: 'center',
                    }}
                  >
                    <Box
                      sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: 2,
                      }}
                    >
                      <Typography
                        sx={{
                          color: '#475569',
                          fontSize: '0.875rem',
                        }}
                      >
                        -
                      </Typography>
                    </Box>
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
