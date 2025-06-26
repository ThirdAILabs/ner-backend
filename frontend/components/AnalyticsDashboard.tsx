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
import { Clock, HardDrive, CheckCircle2 } from 'lucide-react';

import { formatFileSize, formatNumber } from '@/lib/utils';
import type { Tag } from '@/types/analyticsTypes';

interface AnalyticsDashboardProps {
  tokensProcessed: number; // This is actually bytes processed
  tags: Tag[];
  timeTaken: number;
  succeededFileCount: number;
  failedFileCount: number;
  totalFileCount: number;
  dataProcessed: number;
  setTab?: (tab: string) => void;
  setSelectedTag?: (tag: string) => void;
}

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
  tokensProcessed,
  tags,
  timeTaken,
  succeededFileCount,
  failedFileCount,
  totalFileCount,
  dataProcessed,
  setTab,
  setSelectedTag,
}: AnalyticsDashboardProps) {
  const tokenChartData = tags;
  const progress = ((succeededFileCount + failedFileCount) * 100) / totalFileCount || 0;
  const filesSucceeded = (succeededFileCount * 100) / totalFileCount || 0;
  const filesFailed = (failedFileCount * 100) / totalFileCount || 0;

  const formattedTime = formatTime(timeTaken);
  return (
    <div className="space-y-6 w-full">
      {/* Top Widgets */}
      <div className="grid grid-cols-3 gap-4">
        {/* File Progress */}
        <Card className="min-h-[220px] flex flex-col justify-between bg-gradient-to-br from-white to-gray-50/50">
          <CardContent className="flex flex-col flex-1 justify-between pt-4 pb-4">
            <div className="flex items-center space-x-2 mb-4">
              <div className="p-1.5 bg-indigo-100 rounded-lg">
                <CheckCircle2 className="h-4 w-4 text-indigo-600" />
              </div>
              <h3 className="text-sm font-semibold text-gray-700">File Progress</h3>
            </div>

            <div className="flex-1 flex items-center justify-between">
              {/* Circular Progress Chart */}
              <div className="relative h-28 w-28">
                <svg className="h-full w-full -rotate-90 transform" viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="52" fill="none" stroke="#f3f4f6" strokeWidth="6" />
                  <circle
                    cx="60"
                    cy="60"
                    r="52"
                    fill="none"
                    stroke="#16a34a"
                    strokeWidth="6"
                    strokeLinecap="round"
                    strokeDasharray={`${(succeededFileCount / totalFileCount) * 326.725} 326.725`}
                    className="transition-all duration-500 ease-in-out"
                  />
                  {failedFileCount > 0 && (
                    <circle
                      cx="60"
                      cy="60"
                      r="52"
                      fill="none"
                      stroke="#dc2626"
                      strokeWidth="6"
                      strokeLinecap="round"
                      strokeDasharray={`${(failedFileCount / totalFileCount) * 326.725} 326.725`}
                      strokeDashoffset={-((succeededFileCount / totalFileCount) * 326.725)}
                      className="transition-all duration-500 ease-in-out"
                    />
                  )}
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-xl font-bold text-gray-700">{progress.toFixed(0)}%</span>
                  <span className="text-[10px] font-medium text-gray-500">Complete</span>
                </div>
              </div>

              {/* Legend */}
              <div className="flex flex-col space-y-3 pl-2">
                <div className="flex flex-col">
                  <div className="flex items-center space-x-2">
                    <div className="h-2.5 w-2.5 rounded-full bg-green-600"></div>
                    <span className="text-xs font-medium text-gray-600">Succeeded</span>
                  </div>
                  <span className="text-xl font-bold ml-4 text-green-600">
                    {filesSucceeded.toFixed(2)}%
                  </span>
                </div>
                <div className="flex flex-col">
                  <div className="flex items-center space-x-2">
                    <div className="h-2.5 w-2.5 rounded-full bg-red-600"></div>
                    <span className="text-xs font-medium text-gray-600">Failed</span>
                  </div>
                  <span className="text-xl font-bold ml-4 text-red-600">
                    {filesFailed.toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Data Successfully Processed */}
        <Card className="min-h-[220px] flex flex-col justify-between bg-gradient-to-br from-white to-gray-50/50">
          <CardContent className="flex flex-col flex-1 justify-between pt-4 pb-4">
            <div className="flex items-center space-x-2 mb-4">
              <div className="p-1.5 bg-blue-100 rounded-lg">
                <HardDrive className="h-4 w-4 text-blue-600" />
              </div>
              <h3 className="text-sm font-semibold text-gray-700">Data Successfully Processed</h3>
            </div>

            <div className="flex-1 flex flex-col items-center justify-center">
              {dataProcessed !== null && dataProcessed > 0 ? (
                <>
                  <span className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
                    {formatFileSize(dataProcessed)}
                  </span>
                  <div className="flex items-center space-x-2 mt-2">
                    <div className="flex items-center space-x-1 px-2 py-1 bg-green-100 rounded-full">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      <span className="text-sm font-medium text-green-800">Processed</span>
                    </div>
                    <span className="text-base font-medium text-gray-500">
                      of {formatFileSize(tokensProcessed)}
                    </span>
                  </div>
                </>
              ) : (
                <>
                  <span className="text-4xl font-bold tracking-tight bg-gradient-to-r from-gray-600 to-gray-800 bg-clip-text text-transparent">
                    {formatFileSize(tokensProcessed)}
                  </span>
                  <span className="text-sm text-gray-500 mt-2">Total Size</span>
                </>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Time Taken */}
        <Card className="min-h-[220px] flex flex-col justify-between bg-gradient-to-br from-white to-gray-50/50">
          <CardContent className="flex flex-col flex-1 justify-between pt-4 pb-4">
            <div className="flex items-center space-x-2 mb-4">
              <div className="p-1.5 bg-purple-100 rounded-lg">
                <Clock className="h-4 w-4 text-purple-600" />
              </div>
              <h3 className="text-sm font-semibold text-gray-700">
                {progress === 100 ? 'Time Taken' : 'Time Elapsed'}
              </h3>
            </div>

            <div className="flex-1 flex flex-col items-center justify-center">
              <span
                className={`text-4xl font-bold tracking-tight bg-gradient-to-r from-purple-600 to-purple-800 bg-clip-text text-transparent ${timeTakenToTextSize(
                  formattedTime
                )}`}
              >
                {formattedTime}
              </span>

              {/* Ghost element to match height of Data Processed */}
              <span className="text-sm text-transparent mt-2 select-none">
                Processed of {formatFileSize(tokensProcessed)}
              </span>
            </div>
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
                <Bar
                  dataKey="count"
                  fill="#3b82f6"
                  radius={[0, 4, 4, 0]}
                  shape={(props: any) => {
                    const { x, y, width, height, index } = props;
                    return (
                      <g
                        onClick={() => {
                          if (tokenChartData[index].count > 0) {
                            setTab?.('review');
                            setSelectedTag?.(tokenChartData[index].type);
                          }
                        }}
                        style={{ cursor: 'pointer' }}
                      >
                        <rect
                          x={x}
                          y={y}
                          width={width}
                          height={height}
                          fill="#3b82f6"
                          rx="4"
                          ry="4"
                        />
                        <rect x={0} y={y} width="100%" height={height} fill="transparent" />
                      </g>
                    );
                  }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
