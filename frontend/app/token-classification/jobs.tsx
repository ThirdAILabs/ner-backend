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
import { format } from 'date-fns';
import { nerService } from '@/lib/backend';
import { IconButton, Tooltip } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { useHealth } from '@/contexts/HealthProvider';
import { alpha } from '@mui/material/styles';
import { useLicense } from '@/hooks/useLicense';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { useConditionalTelemetry } from '@/hooks/useConditionalTelemetry';
import { isUploadReport } from '@/lib/utils';
import { useEnterprise } from '@/hooks/useEnterprise';

function JobStatus({ report }: { report: ReportWithStatus }) {
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
    reportErrors.length > 0 && reportErrors.includes('license verification failed: quota exceeded');

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
          Failed
        </Typography>
      </Box>
    );
  }

  // Add default values for each status
  const completed = InferenceTaskStatuses?.COMPLETED?.TotalTasks || 0;
  const running = InferenceTaskStatuses?.RUNNING?.TotalTasks || 0;
  const queued = InferenceTaskStatuses?.QUEUED?.TotalTasks || 0;
  const failed = InferenceTaskStatuses?.FAILED?.TotalTasks || 0;
  const aborted = InferenceTaskStatuses?.ABORTED?.TotalTasks || 0;

  const fileCount = report.TotalFileCount || 1;
  const succeededFileCount = report.SucceededFileCount || 0;
  const failedFileCount = report.FailedFileCount || 0;

  const totalTasks = completed + running + queued + failed + aborted;

  function getProgressOutput() {
    if (monthlyQuotaExceeded) {
      return (
        <Box
          component="span"
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            color: 'red',
          }}
        >
          Monthly Quota Exceeded
          <Tooltip title="This scan exceeds the monthly quota for the free-tier. The quota resets on the 1st of each month.">
            <InfoOutlinedIcon fontSize="inherit" sx={{ cursor: 'pointer' }} />
          </Tooltip>
        </Box>
      );
    } else if (totalTasks === 0 && isUploadReport(report)) {
      // Local Upload
      return `Queued...`;
    } else if (totalTasks === 0 && !isUploadReport(report)) {
      // S3 Upload
      return 'Gathering Files...';
    } else if (aborted > 0) {
      let msg = `Scan stopped before completion`;
      if (succeededFileCount > 0) {
        msg += `: ${succeededFileCount}/${fileCount} Processed`;
      }
      if (failedFileCount > 0) {
        msg += `, ${failedFileCount}/${fileCount} Failed`;
      }
      return msg;
    } else if (fileCount > 0 && succeededFileCount === fileCount) {
      return `Files: ${fileCount}/${fileCount} Processed`;
    } else if (fileCount > 0 && failedFileCount === fileCount) {
      return `Files: ${failedFileCount}/${fileCount} Failed`;
    } else if (fileCount > 0) {
      return `Files: ${succeededFileCount}/${fileCount} Processed${failedFileCount > 0 ? `, ${failedFileCount}/${fileCount} Failed` : ''}`;
    }

    return 'Queued...';
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
          {!monthlyQuotaExceeded &&
            succeededFileCount + failedFileCount < fileCount &&
            aborted === 0 && (
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
        {getProgressOutput()}
      </Typography>
    </Box>
  );
}

function LicenseErrorBanner({
  license,
  isEnterprise,
  enterpriseLoading,
  isLicenseValid,
}: {
  license: any;
  isEnterprise: boolean;
  enterpriseLoading: boolean;
  isLicenseValid: boolean;
}) {
  if (!enterpriseLoading && !isLicenseValid && isEnterprise) {
    return (
      <div
        className={`
          px-4 py-3 rounded mb-6 border 
          bg-red-100 border-red-200 text-red-600
        `}
      >
        {license && license.LicenseError === 'expired license'
          ? 'Your license has expired. Please contact ThirdAI support to renew your license.'
          : 'Your license is invalid. Please check your license key or contact ThirdAI support.'}
      </div>
    );
  }
  return null;
}

interface JobProps {
  initialReport: ReportWithStatus;
  onDelete: (reportId: string) => void;
}

function Job({ initialReport, onDelete }: JobProps) {
  const [report, setReport] = useState<ReportWithStatus>({
    ...initialReport,
    isLoadingStatus: true,
  });
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const reportCompletionStatus = async () => {
    try {
      const detailedReport = await nerService.getReport(report.Id);

      setReport((prev) => ({
        ...prev,
        Errors: detailedReport.Errors || [],
        SucceededFileCount: detailedReport.SucceededFileCount,
        FailedFileCount: detailedReport.FailedFileCount,
        TotalFileCount: detailedReport.TotalFileCount,
        detailedStatus: {
          ShardDataTaskStatus: detailedReport.ShardDataTaskStatus,
          InferenceTaskStatuses: detailedReport.InferenceTaskStatuses,
        },
        isLoadingStatus: false,
      }));

      const seenFileCount = detailedReport.SucceededFileCount + detailedReport.FailedFileCount;

      const isCompleted =
        detailedReport.TotalFileCount !== 0 && seenFileCount === detailedReport.TotalFileCount;

      return isCompleted;
    } catch (err) {
      console.error(`Error fetching status for report ${report.Id}:`, err);
      setReport((prev) => ({ ...prev, isLoadingStatus: false }));
      return false;
    }
  };

  const pollReportStatus = async () => {
    const isComplete = await reportCompletionStatus();
    if (isComplete && pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
  };

  useEffect(() => {
    pollReportStatus();
    pollIntervalRef.current = setInterval(pollReportStatus, 5000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const handleDelete = async (reportId: string) => {
    if (window.confirm('Are you sure you want to delete this report?')) {
      try {
        await nerService.deleteReport(reportId);
        onDelete(reportId);
      } catch (error) {
        console.error('Error deleting report:', error);
        // You might want to show an error message to the user
      }
    }
  };

  return (
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
      <TableCell>
        <JobStatus report={report} />
      </TableCell>
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
  );
}

export default function Jobs() {
  const recordEvent = useConditionalTelemetry();
  const [reports, setReports] = useState<ReportWithStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { healthStatus } = useHealth();
  const { license } = useLicense();
  const { isEnterprise, enterpriseLoading } = useEnterprise();
  const [isLicenseValid, setIsLicenseValid] = useState<boolean>(true);

  useEffect(() => {
    recordEvent({
      UserAction: 'View Reports Dashboard',
      UIComponent: 'Reports Dashboard',
      Page: 'Reports Dashboard Page',
    });
  }, []);

  useEffect(() => {
    let licenceValidityCheck = true;
    if (license && license.LicenseError) {
      licenceValidityCheck =
        license.LicenseError !== 'expired license' && license.LicenseError !== 'invalid license';
    }

    setIsLicenseValid(licenceValidityCheck);
  }, [license]);

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [quotaUsedPercentage, setQuotaUsedPercentage] = useState<number | null>(null);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const reportsData = await nerService.listReports();
        reportsData.sort(
          (a, b) => new Date(b.CreationTime).getTime() - new Date(a.CreationTime).getTime()
        );
        setReports(reportsData as ReportWithStatus[]);
      } catch (err) {
        setError('Failed to fetch reports');
        console.error('Error fetching reports:', err);
      } finally {
        setLoading(false);
      }
    };

    // Poll for all reports to stay current on new reports.
    if (healthStatus) {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      fetchReports();
      pollingIntervalRef.current = setInterval(fetchReports, 5000);
    }

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, [healthStatus]);

  useEffect(() => {
    if (license && license?.LicenseInfo?.LicenseType === 'free') {
      setQuotaUsedPercentage(
        (license.LicenseInfo.Usage.UsedBytes / license.LicenseInfo.Usage.MaxBytes) * 100
      );
    }
  }, [reports]);

  if (loading) {
    return (
      <>
        <LicenseErrorBanner
          license={license}
          isEnterprise={isEnterprise}
          enterpriseLoading={enterpriseLoading}
          isLicenseValid={isLicenseValid}
        />
        {typeof quotaUsedPercentage === 'number' && quotaUsedPercentage > 75 && (
          <div
            className={`
                      px-4 py-3 rounded mb-6 border 
                      bg-yellow-100 border-yellow-200 text-yellow-600
                    `}
          >
            You have used {quotaUsedPercentage?.toFixed(2)}% of your monthly quota. Any report
            exceeding the quota will not be processed. The quota resets on the 1st of each month.
          </div>
        )}
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
              {healthStatus && isLicenseValid ? (
                <Link href={`/token-classification/jobs/new`} passHref legacyBehavior>
                  <a style={{ textDecoration: 'none' }}>
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
                    >
                      New Scan
                    </Button>
                  </a>
                </Link>
              ) : (
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<Plus size={20} />}
                  disabled
                  sx={{
                    bgcolor: '#2563eb',
                    textTransform: 'none',
                    fontWeight: 500,
                    px: 3,
                    py: 1.5,
                    borderRadius: '8px',
                    boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                  }}
                >
                  New Scan
                </Button>
              )}
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
                    reports.map((report) => (
                      <Job key={report.Id} initialReport={report} onDelete={handleDelete} />
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
      </>
    );
  }

  if (error) {
    return (
      <>
        <LicenseErrorBanner
          license={license}
          isEnterprise={isEnterprise}
          enterpriseLoading={enterpriseLoading}
          isLicenseValid={isLicenseValid}
        />

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
            <Typography sx={{ textAlign: 'center', py: 2, color: 'error.main' }}>
              {error}
            </Typography>
          </CardContent>
        </Card>
      </>
    );
  }

  const handleDelete = async (reportId: string) => {
    setReports(reports.filter((report) => report.Id !== reportId));
  };

  return (
    <>
      <LicenseErrorBanner
        license={license}
        isEnterprise={isEnterprise}
        enterpriseLoading={enterpriseLoading}
        isLicenseValid={isLicenseValid}
      />

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
                  reports.map((report) => (
                    <Job key={report.Id} initialReport={report} onDelete={handleDelete} />
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
    </>
  );
}
