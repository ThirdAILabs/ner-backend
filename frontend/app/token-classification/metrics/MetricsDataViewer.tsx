'use client';

import React, { useEffect, useState } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { nerService } from '@/lib/backend';
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
import type { SavedFeedback, FinetuneRequest } from '@/lib/backend';
import { Toaster, toast } from 'react-hot-toast';

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
  const [feedbackData, setFeedbackData] = useState<SavedFeedback[]>([]);
  const [loadingFeedback, setLoadingFeedback] = useState(false);

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

  const handleDeleteFeedback = async (id: string) => {
    if (!modelId) return;
    try {
      await nerService.deleteModelFeedback(modelId, id);
      setFeedbackData(feedbackData.filter((feedback) => feedback.id !== id));
    } catch (error) {
      console.error('Failed to delete feedback:', error);
    }
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
    </Box>
  );
};

export default MetricsDataViewer;