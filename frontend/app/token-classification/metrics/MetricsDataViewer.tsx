'use client';

import React, { useEffect, useState } from 'react';
import { Box, Grid, Card, CardContent, Typography, CircularProgress } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, LabelList } from 'recharts';
import { nerService, InferenceMetrics, ThroughputMetrics } from '@/lib/backend';

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

  useEffect(() => {
    let mounted = true;
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
        } else {
          setTpMetrics(null);
        }

        // 3) build inference‐series by calling summary for each day = 1..days
        // const infPromises = Array.from({ length: days }, (_, i) =>
        //   nerService.getInferenceMetrics(modelId, i + 1)
        // );
        // const infResults = await Promise.all(infPromises);
        // if (!mounted) return;
        // setInfSeries(
        //   infResults.map((res, i) => ({
        //     day: i + 1,
        //     dataMB: parseFloat(res.DataProcessedMB.toFixed(2)),
        //     tokens: res.TokensProcessed
        //   }))
        // );
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
      <div className="space-y-6 w-full">
        <div className="grid grid-cols-5 gap-4">
          {/* In-Progress Tasks */}
          <Card className="flex flex-col justify-between">
            <CardContent className="flex flex-col items-center justify-center flex-1 pt-6">
              <div className="relative h-32 w-32">
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-gray-700">{infMetrics.InProgress}</span>
                </div>
              </div>
              <h3 className="mt-auto text-sm text-muted-foreground">In-Progress Files</h3>
            </CardContent>
          </Card>

          {/* Completed Tasks */}
          <Card className="flex flex-col justify-between">
            <CardContent className="flex flex-col items-center justify-center flex-1 pt-6">
              <div className="relative h-32 w-32">
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-gray-700">{infMetrics.Completed}</span>
                </div>
              </div>
              <h3 className="mt-auto text-sm text-muted-foreground">Completed Files</h3>
            </CardContent>
          </Card>

          {/* Throughput */}
          <Card className="flex flex-col justify-between">
            <CardContent className="flex flex-col items-center justify-center flex-1 pt-6">
              <div className="relative h-32 w-32">
                <div className="absolute inset-0 flex items-center justify-center">
                  <span
                    className={`font-bold text-gray-700 text-center ${
                      tpMetrics?.ThroughputMBPerHour &&
                      (tpMetrics?.ThroughputMBPerHour > 1000
                        ? 'text-xl'
                        : tpMetrics?.ThroughputMBPerHour > 100
                          ? 'text-2xl'
                          : 'text-3xl')
                    }`}
                  >
                    {tpMetrics?.ThroughputMBPerHour == null
                      ? '-'
                      : `${tpMetrics?.ThroughputMBPerHour.toFixed(2).toLocaleString()} MB/Hour`}
                  </span>
                </div>
              </div>
              <h3 className="mt-auto text-sm text-muted-foreground">Throughput</h3>
            </CardContent>
          </Card>

          {/* Data Processed */}
          <Card className="flex flex-col justify-between">
            <CardContent className="flex flex-col items-center justify-center flex-1 pt-6">
              <div className="relative h-32 w-32">
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-gray-700">
                    {infMetrics.DataProcessedMB.toFixed(2).toLocaleString()} MB
                  </span>
                </div>
              </div>
              <h3 className="mt-auto text-sm text-muted-foreground">Data Processed</h3>
            </CardContent>
          </Card>

          {/* Tokens Processed */}
          <Card className="flex flex-col justify-between">
            <CardContent className="flex flex-col items-center justify-center flex-1 pt-6">
              <div className="relative h-32 w-32">
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold text-gray-700">
                    {infMetrics.TokensProcessed.toLocaleString()}
                  </span>
                </div>
              </div>
              <h3 className="mt-auto text-sm text-muted-foreground">Tokens Processed</h3>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
};

export default MetricsDataViewer;
