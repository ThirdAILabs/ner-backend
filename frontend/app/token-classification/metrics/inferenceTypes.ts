export interface InferenceMetrics {
  InProgress: number;
  Completed: number;
  Failed: number;
  DataProcessedMB: number;
  TokensProcessed: number;
}

export interface ThroughputMetrics {
  ThroughputMBPerHour?: number;
}
