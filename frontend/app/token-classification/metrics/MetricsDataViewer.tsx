'use client';

import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LabelList
} from 'recharts';
import {
  nerService,
  InferenceMetrics,
  ThroughputMetrics
} from '@/lib/backend';

interface MetricsDataViewerProps {
  modelId?: string;
  days: number;
}

const MetricsDataViewer: React.FC<MetricsDataViewerProps> = ({
  modelId,
  days
}) => {
  const [infMetrics, setInfMetrics] = useState<InferenceMetrics | null>(
    null
  );
  const [tpMetrics, setTpMetrics] = useState<ThroughputMetrics | null>(
    null
  );
  const [infSeries, setInfSeries] = useState<
    { day: number; dataMB: number; tokens: number }[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        // 1) fetch the summary for the full window
        const summary = await nerService.getInferenceMetrics(
          modelId,
          days
        );
        if (!mounted) return;
        setInfMetrics(summary);

        // 2) fetch throughput summary card
        if (modelId) {
          const tp = await nerService.getThroughputMetrics(
            modelId
          );
          if (!mounted) return;
          setTpMetrics(tp);
        } else {
          setTpMetrics(null);
        }

        // 3) build inference‐series by calling summary for each day = 1..days
        const infPromises = Array.from({ length: days }, (_, i) =>
          nerService.getInferenceMetrics(modelId, i + 1)
        );
        const infResults = await Promise.all(infPromises);
        if (!mounted) return;
        setInfSeries(
          infResults.map((res, i) => ({
            day: i + 1,
            dataMB: parseFloat(res.DataProcessedMB.toFixed(2)),
            tokens: res.TokensProcessed
          }))
        );
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
  }, [modelId, days]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  if (!infMetrics) {
    return null;
  }

  return (
    <>
      {/* 1) Summary cards */}
      <Grid container spacing={2} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2">
                Completed Tasks
              </Typography>
              <Typography variant="h5">
                {infMetrics.Completed}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2">
                In-Progress
              </Typography>
              <Typography variant="h5">
                {infMetrics.InProgress}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2">
                Data Processed (MB)
              </Typography>
              <Typography variant="h5">
                {infMetrics.DataProcessedMB.toFixed(2)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="subtitle2">
                Tokens Processed
              </Typography>
              <Typography variant="h5">
                {infMetrics.TokensProcessed}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        {tpMetrics && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="subtitle2">
                  Throughput (MB/hr)
                </Typography>
                <Typography variant="h4">
                  {tpMetrics.ThroughputMBPerHour.toFixed(2)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </>
  );
};

export default MetricsDataViewer;
