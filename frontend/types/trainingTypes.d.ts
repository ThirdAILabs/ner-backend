// Training types
interface LabelMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

interface ExampleCategories {
  true_positives: TrainingExample[];
  false_positives: TrainingExample[];
  false_negatives: TrainingExample[];
}

interface TrainingExample {
  id: string;
  text: string;
  tokens: string[];
  labels: string[];
  predictions: string[];
}

interface TrainReportData {
  timestamp: string;
  duration: number;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    label_metrics: Record<string, LabelMetrics>;
  };
  examples: ExampleCategories;
}
