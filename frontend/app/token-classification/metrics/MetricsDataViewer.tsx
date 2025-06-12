'use client';

import React, { useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography, CircularProgress } from '@mui/material';
import { nerService } from '@/lib/backend';
import type { InferenceMetrics, ThroughputMetrics } from './inferenceTypes';
import { formatFileSize, formatNumber } from '@/lib/utils';
import { useHealth } from '@/contexts/HealthProvider';
import MetricsDataViewerCard from '@/components/ui/MetricsDataViewerCard';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import type { Feedback, FinetuneRequest } from '@/lib/backend';

interface MetricsDataViewerProps {
  modelId?: string;
  days: number;
}

const MetricsDataViewer: React.FC<MetricsDataViewerProps> = ({ modelId, days }) => {
  const [infMetrics, setInfMetrics] = useState<InferenceMetrics | null>(null);
  const [tpMetrics, setTpMetrics] = useState<ThroughputMetrics | null>(null);
  const [infSeries, setInfSeries] = useState<{ day: number; dataMB: number; tokens: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [throughput, setThroughput] = useState<string | null>('-');
  const { healthStatus } = useHealth();

  // State for feedback data
  const [feedbackData, setFeedbackData] = useState<Feedback[]>([]);
  const [loadingFeedback, setLoadingFeedback] = useState(false);

  // State for finetuning
  const [showFinetuneDialog, setShowFinetuneDialog] = useState(false);
  const [finetuneModelName, setFinetuneModelName] = useState('');
  const [finetuneTaskPrompt, setFinetuneTaskPrompt] = useState('');
  const [finetuning, setFinetuning] = useState(false);

  function getFontSize(value: string) {
    if (!value) return '2rem';
    if (value.length > 7) return '1.25rem';
    if (value.length > 5) return '1.5rem';
    if (value.length > 3) return '1.75rem';

    return '2rem';
  }

  useEffect(() => {
    let mounted = true;

    // Don't make API calls if health check hasn't passed
    if (!healthStatus) {
      return;
    }

    setLoading(true);
    setError(null);

    (async () => {
      try {
        // 1) fetch the summary for the full window
        const summary = await nerService.getInferenceMetrics(modelId, days);
        if (!mounted) return;
        setInfMetrics(summary);

        // 2) fetch throughput summary card
        if (modelId) {
          const tp = await nerService.getThroughputMetrics(modelId);
          if (!mounted) return;
          setTpMetrics(tp);
          setThroughput(
            tp.ThroughputMBPerHour !== undefined
              ? formatFileSize(tp.ThroughputMBPerHour, true)
              : '-'
          );
        } else {
          setTpMetrics(null);
        }

        // 3) Fetch feedback data if modelId is provided
        if (modelId) {
          setLoadingFeedback(true);
          try {
            const feedback = await nerService.getFeedbackSamples(modelId);
            if (!mounted) return;
            setFeedbackData(feedback);
          } catch (e: any) {
            console.error('Failed to load feedback data:', e);
            // Don't set error for feedback, just log it
          } finally {
            if (mounted) setLoadingFeedback(false);
          }
        }
      } catch (e: any) {
        if (!mounted) return;
        setError(e.message || 'Failed to load metrics');
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [modelId, days, healthStatus]);

  // Generate consistent colors for different tag types
  const getTagColors = (label: string) => {
    if (label === 'O') {
      return null; // No highlighting for 'O' tags
    }

    // Color palette for different tag types
    const colorPalette = [
      { text: '#FFE8E8', tag: '#FF6B6B' }, // Red
      { text: '#E8F4FF', tag: '#4A90E2' }, // Blue
      { text: '#E8FFE8', tag: '#51C878' }, // Green
      { text: '#FFF8E8', tag: '#F39C12' }, // Orange
      { text: '#F0E8FF', tag: '#9B59B6' }, // Purple
      { text: '#E8FFFF', tag: '#1ABC9C' }, // Teal
      { text: '#FFE8F8', tag: '#E91E63' }, // Pink
      { text: '#F8FFE8', tag: '#8BC34A' }, // Light Green
      { text: '#E8E8FF', tag: '#6366F1' }, // Indigo
      { text: '#FFF0E8', tag: '#FF8C00' }, // Dark Orange
    ];

    // Create a simple hash function for consistent color assignment
    let hash = 0;
    for (let i = 0; i < label.length; i++) {
      hash = label.charCodeAt(i) + ((hash << 5) - hash);
    }
    const colorIndex = Math.abs(hash) % colorPalette.length;

    return colorPalette[colorIndex];
  };

  const renderHighlightedToken = (token: string, label: string) => {
    const tagColors = getTagColors(label);

    if (!tagColors) {
      return token; // Return plain token for 'O' tags
    }

    return (
      <span>
        <span
          style={{
            backgroundColor: tagColors.text,
            padding: '2px 4px',
            borderRadius: '2px',
            userSelect: 'none',
            display: 'inline-flex',
            alignItems: 'center',
            wordBreak: 'break-word',
          }}
        >
          {token}
          <span
            style={{
              backgroundColor: tagColors.tag,
              color: 'white',
              fontSize: '11px',
              fontWeight: 'bold',
              borderRadius: '2px',
              marginLeft: '4px',
              padding: '1px 3px',
            }}
          >
            {label}
          </span>
        </span>
      </span>
    );
  };

  const handleFinetuneClick = () => {
    if (!modelId) return;

    // Generate a default name based on the current model
    const timestamp = new Date().toISOString().slice(0, 16).replace('T', '_').replace(':', '-');
    setFinetuneModelName(`finetuned_${timestamp}`);
    setFinetuneTaskPrompt('');
    setShowFinetuneDialog(true);
  };

  const handleFinetuneSubmit = async () => {
    if (!modelId || !finetuneModelName.trim()) return;

    setFinetuning(true);
    try {
      const request: FinetuneRequest = {
        name: finetuneModelName.trim(),
        task_prompt: finetuneTaskPrompt.trim() || undefined,
        samples: feedbackData.length > 0 ? feedbackData : undefined,
      };

      const response = await nerService.finetuneModel(modelId, request);
      console.log('Finetuning started for new model:', response.ModelId);

      // Close dialog and reset state
      setShowFinetuneDialog(false);
      setFinetuneModelName('');
      setFinetuneTaskPrompt('');

      // You could show a success message or redirect to the new model
      alert(`Finetuning started successfully! New model ID: ${response.ModelId}`);
    } catch (error) {
      console.error('Finetuning failed:', error);
      // Error handling is already done in the service layer
    } finally {
      setFinetuning(false);
    }
  };

  const handleFinetuneCancel = () => {
    setShowFinetuneDialog(false);
    setFinetuneModelName('');
    setFinetuneTaskPrompt('');
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography sx={{ color: 'error.main', fontSize: '0.875rem' }}>{error}</Typography>
      </Box>
    );
  }

  if (!infMetrics) {
    return null;
  }

  return (
    <>
      <Box sx={{ p: 3 }}>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: 'repeat(5, 1fr)',
            gap: 3,
          }}
        >
          {/* In-Progress Tasks */}
          <MetricsDataViewerCard value={infMetrics.InProgress} label="In-Progress Scans" />

          {/* Completed Tasks */}
          <MetricsDataViewerCard
            value={infMetrics.Completed + infMetrics.Failed}
            label="Completed Scans"
          />

          {/* Throughput */}
          <MetricsDataViewerCard
            value={throughput === '-' ? '-' : `${throughput}/Hour`}
            label="Throughput"
          />

          {/* Data Processed */}
          <MetricsDataViewerCard
            value={formatFileSize(infMetrics.DataProcessedMB * 1024 * 1024)}
            label="Data Processed"
          />

          {/* Tokens Processed */}
          <MetricsDataViewerCard
            value={formatNumber(infMetrics.TokensProcessed)}
            label="Tokens Processed"
          />
        </Box>

        {/* Fine-tuned Feedback Data */}
        {modelId && (
          <Box sx={{ mt: 4 }}>
            <Box
              sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}
            >
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  fontSize: '1.25rem',
                  color: '#4a5568',
                }}
              >
                User Feedback
              </Typography>
              {feedbackData.length > 0 && (
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleFinetuneClick}
                  disabled={loadingFeedback}
                  sx={{
                    textTransform: 'none',
                    fontWeight: 600,
                    px: 3,
                  }}
                >
                  Finetune Model
                </Button>
              )}
            </Box>
            <div className="border rounded-md">
              {loadingFeedback ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                  <CircularProgress size={20} />
                </Box>
              ) : feedbackData.length === 0 ? (
                <Box sx={{ p: 4, textAlign: 'center' }}>
                  <Typography sx={{ color: 'text.secondary', fontSize: '0.875rem' }}>
                    No feedback data available for this model
                  </Typography>
                </Box>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Feedback Text</TableHead>
                      <TableHead className="w-[50px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {feedbackData.map((feedback, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          {feedback.tokens.map((token, tokenIndex) => (
                            <span key={tokenIndex}>
                              {renderHighlightedToken(token, feedback.labels[tokenIndex])}
                              {tokenIndex < feedback.tokens.length - 1 ? ' ' : ''}
                            </span>
                          ))}
                        </TableCell>
                        <TableCell className="text-right">
                          <button
                            className="text-gray-700 hover:text-gray-700 transition-colors"
                            onClick={() => {
                              // TODO: Implement delete functionality when backend is ready
                              console.log('Delete feedback:', feedback);
                            }}
                            title="Delete feedback"
                          >
                            ✕
                          </button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </div>
          </Box>
        )}
      </Box>

      {/* Finetune Dialog */}
      <Dialog open={showFinetuneDialog} onClose={handleFinetuneCancel} maxWidth="sm" fullWidth>
        <DialogTitle>Finetune Model</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <Typography variant="body2" sx={{ mb: 3, color: 'text.secondary' }}>
              Create a new finetuned model using the feedback data you've reviewed. This will use
              all {feedbackData.length} feedback samples as training data.
            </Typography>

            <TextField
              autoFocus
              margin="dense"
              label="Model Name"
              fullWidth
              variant="outlined"
              value={finetuneModelName}
              onChange={(e) => setFinetuneModelName(e.target.value)}
              helperText="Enter a name for the new finetuned model"
              sx={{ mb: 2 }}
              required
            />

            <TextField
              margin="dense"
              label="Task Prompt (Optional)"
              fullWidth
              multiline
              rows={3}
              variant="outlined"
              value={finetuneTaskPrompt}
              onChange={(e) => setFinetuneTaskPrompt(e.target.value)}
              helperText="Optional custom prompt to guide the finetuning process"
              placeholder="e.g., Focus on improving accuracy for person names and locations..."
            />
          </Box>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button
            onClick={handleFinetuneCancel}
            disabled={finetuning}
            sx={{ textTransform: 'none' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleFinetuneSubmit}
            variant="contained"
            disabled={finetuning || !finetuneModelName.trim()}
            sx={{ textTransform: 'none', ml: 1 }}
          >
            {finetuning ? (
              <>
                <CircularProgress size={16} sx={{ mr: 1 }} />
                Starting Finetuning...
              </>
            ) : (
              'Start Finetuning'
            )}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default MetricsDataViewer;
