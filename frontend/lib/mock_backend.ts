import type { TrainReportData, LabelMetrics, ExampleCategories, TrainingExample } from '@/lib/types';
import { useState, useEffect } from 'react';

// Mock data for training reports
const mockMetrics: Record<string, LabelMetrics> = {
  'O': {
    precision: 0.95,
    recall: 0.92,
    f1: 0.93,
    support: 100
  },
  'NAME': {
    precision: 0.88,
    recall: 0.85,
    f1: 0.86,
    support: 50
  },
  'PHONE': {
    precision: 0.91,
    recall: 0.89,
    f1: 0.90,
    support: 30
  },
  'EMAIL': {
    precision: 0.94,
    recall: 0.93,
    f1: 0.94,
    support: 25
  },
  'ADDRESS': {
    precision: 0.87,
    recall: 0.86,
    f1: 0.87,
    support: 40
  }
};

const mockExamples: ExampleCategories = {
  true_positives: [
    { id: "1", text: "John Smith", tokens: ["John", "Smith"], labels: ["NAME", "NAME"], predictions: ["NAME", "NAME"] },
    { id: "2", text: "555-123-4567", tokens: ["555-123-4567"], labels: ["PHONE"], predictions: ["PHONE"] },
    { id: "3", text: "john.smith@example.com", tokens: ["john.smith@example.com"], labels: ["EMAIL"], predictions: ["EMAIL"] }
  ],
  false_positives: [
    { id: "4", text: "John", tokens: ["John"], labels: ["O"], predictions: ["NAME"] },
    { id: "5", text: "123-4567", tokens: ["123-4567"], labels: ["O"], predictions: ["PHONE"] }
  ],
  false_negatives: [
    { id: "6", text: "Michael Brown", tokens: ["Michael", "Brown"], labels: ["NAME", "NAME"], predictions: ["O", "O"] },
    { id: "7", text: "555-987-6543", tokens: ["555-987-6543"], labels: ["PHONE"], predictions: ["O"] }
  ]
};

const mockTrainReport: TrainReportData = {
  timestamp: "2023-10-05T12:00:00Z",
  duration: 3600,
  metrics: {
    accuracy: 0.90,
    precision: 0.88,
    recall: 0.86,
    f1: 0.87,
    label_metrics: mockMetrics
  },
  examples: mockExamples
};

// Mock implementation of retrainTokenClassifier
export async function retrainTokenClassifier({
  model_name,
  base_model_id,
}: {
  model_name: string;
  base_model_id: string;
}): Promise<any> {
  console.log(`Mock: Retraining token classifier with model_name: ${model_name}, base_model_id: ${base_model_id}`);
  
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  return {
    status: 'success',
    message: 'Model retraining initiated successfully',
    model_id: `mock-${Date.now()}`,
    model_name: model_name
  };
}

// Mock implementation of trainUDTWithCSV
export function trainUDTWithCSV({
  model_name,
  file,
  base_model_identifier,
  test_split = 0.1,
}: {
  model_name: string;
  file: File;
  base_model_identifier: string;
  test_split?: number;
}): Promise<any> {
  console.log(`Mock: Training UDT with CSV - model_name: ${model_name}, base_model_identifier: ${base_model_identifier}, test_split: ${test_split}`);
  console.log(`File details: ${file.name}, ${file.size} bytes, ${file.type}`);
  
  // Simulate API delay
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        status: 'success',
        message: 'Model training with CSV initiated successfully',
        model_id: `mock-csv-${Date.now()}`,
        model_name: model_name
      });
    }, 1000);
  });
}

// Mock implementation of getTrainReport
export async function getTrainReport(modelId: string): Promise<{ data: TrainReportData }> {
  console.log(`Mock: Getting training report for model ID: ${modelId}`);
  
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 800));
  
  return {
    data: mockTrainReport
  };
}

// Mock implementation of useLabels hook
export function useLabels({
  deploymentUrl,
  maxRecentLabels = 5,
}: {
  deploymentUrl: string;
  maxRecentLabels?: number;
}): {
  allLabels: Set<string>;
  recentLabels: string[];
  error: Error | null;
  isLoading: boolean;
  refresh: () => Promise<void>;
} {
  const [allLabels, setAllLabels] = useState<Set<string>>(new Set());
  const [recentLabels, setRecentLabels] = useState<string[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchLabels = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Mock labels
      const mockLabels = ['NAME', 'PHONE', 'EMAIL', 'ADDRESS', 'O', 'DATE', 'LOCATION', 'ORGANIZATION'];
      setAllLabels(new Set(mockLabels));
      setRecentLabels(mockLabels.slice(0, maxRecentLabels));
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch labels'));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchLabels();
  }, [deploymentUrl, maxRecentLabels]);

  return {
    allLabels,
    recentLabels,
    error,
    isLoading,
    refresh: fetchLabels
  };
}

// Mock implementation of useRecentSamples hook
export function useRecentSamples({
  deploymentUrl,
  maxRecentSamples = 5,
}: {
  deploymentUrl: string;
  maxRecentSamples?: number;
}): {
  recentSamples: { tokens: string[]; tags: string[] }[];
  error: Error | null;
  isLoading: boolean;
  refresh: () => Promise<void>;
} {
  const [recentSamples, setRecentSamples] = useState<{ tokens: string[]; tags: string[] }[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchSamples = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Mock samples
      const mockSamples = [
        { 
          tokens: ['John', 'Smith', 'contacted', 'us', 'at', '555-123-4567'], 
          tags: ['NAME', 'NAME', 'O', 'O', 'O', 'PHONE'] 
        },
        { 
          tokens: ['Please', 'email', 'Jane', 'Doe', 'at', 'jane.doe@example.com'], 
          tags: ['O', 'O', 'NAME', 'NAME', 'O', 'EMAIL'] 
        },
        { 
          tokens: ['Our', 'office', 'is', 'located', 'at', '123', 'Main', 'St'], 
          tags: ['O', 'O', 'O', 'O', 'O', 'ADDRESS', 'ADDRESS', 'ADDRESS'] 
        },
        { 
          tokens: ['Robert', 'Johnson', 'will', 'visit', 'on', 'January', '15th'], 
          tags: ['NAME', 'NAME', 'O', 'O', 'O', 'DATE', 'DATE'] 
        },
        { 
          tokens: ['The', 'meeting', 'is', 'at', 'Google', 'headquarters'], 
          tags: ['O', 'O', 'O', 'O', 'ORGANIZATION', 'O'] 
        }
      ];
      
      setRecentSamples(mockSamples.slice(0, maxRecentSamples));
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch recent samples'));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSamples();
  }, [deploymentUrl, maxRecentSamples]);

  return {
    recentSamples,
    error,
    isLoading,
    refresh: fetchSamples
  };
}

// Mock implementation of getLabels
export async function getLabels(): Promise<string[]> {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 500));
  
  // Return a predefined set of labels
  return [
    'NAME', 
    'PHONE', 
    'EMAIL', 
    'ADDRESS', 
    'DATE', 
    'ORGANIZATION', 
    'LOCATION', 
    'O'
  ];
} 