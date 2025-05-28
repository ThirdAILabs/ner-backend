import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import _ from 'lodash';

interface ClusterSpecs {
  cpus: number;
  vendorId: string;
  modelName: string;
  cpuMhz: number;
}

export interface Tag {
  type: string;
  count: number;
}
interface AnalyticsDashboardProps {
  progress: number;
  tokensProcessed: number; // This is actually bytes processed
  tags: Tag[];
  timeTaken: number;
  succeededFileCount: number;
  failedFileCount: number;
  totalFileCount: number;
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Format number for token counts
const formatNumber = (num: number): string => {
  if (num >= 1000000) {
    return `${(num / 1000000).toFixed(2)}M`;
  }
  if (num >= 1000) {
    return `${(num / 1000).toFixed(1)}K`;
  }
  return num.toString();
};

const formatTime = (time: number): string => {
  if (time < 60) return `${time.toFixed(2)}s`;

  const minutes = Math.floor(time / 60);
  if (minutes < 60) {
    const remainingSeconds = time % 60;
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds.toFixed(0)}s` : `${minutes}m`;
  }

  const hours = Math.floor(time / 3600);
  const remainingMinutes = Math.floor((time % 3600) / 60);
  return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
};

const timeTakenToTextSize = (timeTaken: string) => {
  if (timeTaken.length > 8) {
    return 'text-2xl';
  }
  if (timeTaken.length > 6) {
    return 'text-3xl';
  }
  return 'text-4xl';
};

export function AnalyticsDashboard({
  progress,
  tokensProcessed,
  tags,
  timeTaken,
  succeededFileCount,
  failedFileCount,
  totalFileCount,
}: AnalyticsDashboardProps) {
  const tokenChartData = tags;
  const filesSucceeded = ((succeededFileCount * 100) / totalFileCount).toFixed(0) || 0;
  const filesFailed = ((failedFileCount * 100) / totalFileCount).toFixed(0) || 0;

  const formattedTime = formatTime(timeTaken);

  return (
    <div className="space-y-6 w-full">
      {/* Top Widgets */}
      <div className="grid grid-cols-3 gap-4">
        {/* Progress Widget */}
        <Card className="col-start-1 flex flex-col justify-between">
          <CardContent className="flex flex-col items-center justify-center flex-1 pt-6">
            <div className="relative h-36 w-36">
              <svg className="h-full w-full" viewBox="0 0 120 120">
                {/* Background circle */}
                {/* <circle cx="60" cy="60" r="48" fill="none" stroke="#facc15" strokeWidth="10" /> */}
                <circle cx="60" cy="60" r="48" fill="none" stroke="#dddddd" strokeWidth="10" />

                {/* Success arc (green) */}
                <circle
                  cx="60"
                  cy="60"
                  r="48"
                  fill="none"
                  stroke="#4caf50"
                  strokeWidth="10"
                  strokeDasharray={`${(succeededFileCount / totalFileCount) * 301.592} 301.592`}
                  transform="rotate(-90 60 60)"
                />

                {/* Failure arc (red), offset by the success arc */}
                <circle
                  cx="60"
                  cy="60"
                  r="48"
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth="10"
                  strokeDasharray={`${(failedFileCount / totalFileCount) * 301.592} 301.592`}
                  strokeDashoffset={-((succeededFileCount / totalFileCount) * 301.592)}
                  transform="rotate(-90 60 60)"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center space-y-0">
                <span className="text-xl font-bold text-gray-700">{progress}%</span>
                <span className="text-xs  text-gray-400">{filesSucceeded}% processed</span>
                <span className="text-xs  text-gray-400">{filesFailed}% failed</span>
              </div>
            </div>
            <h3 className="mt-auto text-sm text-muted-foreground">Progress</h3>
          </CardContent>
        </Card>

        {/* Data Processed Widget (formerly Tokens Processed) */}
        <Card className="flex flex-col justify-between">
          <CardContent className="flex flex-col items-center pt-6 h-full">
            <div className="flex-1 flex items-center">
              <span className="text-4xl font-semibold text-gray-700 text-center">
                {formatFileSize(tokensProcessed)}
              </span>
            </div>
            <h3 className="text-sm text-muted-foreground">Data Processed</h3>
          </CardContent>
        </Card>

        {/* Time Taken */}
        <Card className="flex flex-col justify-between">
          <CardContent className="flex flex-col items-center pt-6 h-full">
            <div className="flex-1 flex items-center">
              <span
                className={`font-semibold text-gray-700 text-center ${timeTakenToTextSize(
                  formattedTime
                )}`}
              >
                {formattedTime}
              </span>
            </div>
            <h3 className="text-sm text-muted-foreground">Time Taken</h3>
          </CardContent>
        </Card>
      </div>

      {/* Token Distribution Chart */}
      <Card>
        <CardHeader>
          {/* <CardTitle>Identified Tokens</CardTitle> */}
          <div className="text-2xl font-semibold text-gray-700">Identified Tokens</div>
        </CardHeader>
        <CardContent>
          <div style={{ height: `${Math.max(300, tokenChartData.length * 50)}px` }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={tokenChartData}
                layout="vertical"
                margin={{ top: 20, right: 30, left: 120, bottom: 30 }}
                barSize={30}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  label={{
                    value: 'Number of tokens',
                    position: 'bottom',
                    offset: 15,
                  }}
                  tickFormatter={formatNumber}
                />
                <YAxis dataKey="type" type="category" />
                <Tooltip
                  formatter={(value: number) => formatNumber(value)}
                  labelFormatter={(label) => `Type: ${label}`}
                />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
