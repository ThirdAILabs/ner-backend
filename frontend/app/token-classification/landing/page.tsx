'use client';

import { Suspense } from 'react';
import { useEffect, useState } from 'react';
import { Tabs, Tab, Box, CircularProgress, Typography, Link as MuiLink } from '@mui/material'; // Added CircularProgress
import * as _ from 'lodash';
import { useParams, useSearchParams } from 'next/navigation';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

// Import our implemented components
// import Interact from './interact';
import Dashboard from '../dashboard';
import Jobs from '../jobs';

interface ModelUpdateProps {
  username: string;
  modelName: string;
  deploymentUrl: string;
  workflowNames: string[];
  deployStatus: string;
  modelId: string;
}

type LabelMetrics = {
  [key: string]: {
    precision: number;
    recall: number;
    fmeasure: number;
  };
};

type ExampleCategories = {
  true_positives: Record<string, any>;
  false_positives: Record<string, any>;
  false_negatives: Record<string, any>;
};

type TrainReportData = {
  before_train_metrics: LabelMetrics;
  after_train_metrics: LabelMetrics;
  after_train_examples: ExampleCategories;
};

const emptyMetrics: LabelMetrics = {
  O: { precision: 0, recall: 0, fmeasure: 0 },
};

const emptyExamples: ExampleCategories = {
  true_positives: {},
  false_positives: {},
  false_negatives: {},
};

const emptyReport: TrainReportData = {
  before_train_metrics: emptyMetrics,
  after_train_metrics: emptyMetrics,
  after_train_examples: emptyExamples,
};

const getTrainReport = async (workflowName: string) => {
  return { data: emptyReport };
};

// --- Renamed original Page component to PageContents ---
function PageContents() {
  const params = useParams();
  const workflowName = (params.deploymentId as string) || 'PII';
  const searchParams = useSearchParams();
  const defaultTab = searchParams.get('tab') || 'jobs';
  const [tabValue, setTabValue] = useState(defaultTab);
  const [trainReport, setTrainReport] = useState<TrainReportData>(emptyReport);
  const [isLoadingReport, setIsLoadingReport] = useState(false);
  const [reportError, setReportError] = useState('');

  // Update tabValue if searchParams change after initial load (e.g., browser back/forward)
  useEffect(() => {
    setTabValue(searchParams.get('tab') || 'jobs');
  }, [searchParams]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: string) => {
    setTabValue(newValue);
  };

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setIsLoadingReport(true);
        setReportError('');
        const response = await getTrainReport(workflowName);
        setTrainReport(response.data);
      } catch (error) {
        setReportError(error instanceof Error ? error.message : 'Failed to fetch training report');
        setTrainReport(emptyReport);
      } finally {
        setIsLoadingReport(false);
      }
    };

    fetchReport();
  }, [workflowName]);

  // Mock data for ModelUpdateProps (not used in current JSX, consider removing if unused)
  const mockModelData: ModelUpdateProps = {
    username: 'testuser',
    modelName: 'token-classifier',
    deploymentUrl: 'https://mock-deployment-url.com',
    workflowNames: ['token-classifier', 'token-classifier-v2'],
    deployStatus: 'complete',
    modelId: 'mock-model-id-123',
  };

  return (
    <div style={{ width: '75%', minHeight: '100vh', margin: '0 auto' }}>
      <header
        style={{
          width: '100%',
          padding: '16px',
          borderBottom: '1px solid #e0e0e0',
          display: 'flex',
          flexDirection: 'row',
        }}
      >
        <Button variant="outline" size="sm" asChild>
          <Link href={`/`} className="flex items-center">
            <ArrowLeft className="mr-1 h-4 w-4" /> Back
          </Link>
        </Button>
        <div
          style={{
            maxWidth: '1200px',
            margin: '0 auto',
            marginBottom: '8px',
            display: 'flex',
            flexDirection: 'row',
            gap: '20px',
            marginLeft: '30%',
          }}
        >
          <Image
            src="/thirdai-logo.png"
            alt="ThirdAI Logo"
            width={40}
            height={40}
            style={{ objectFit: 'contain' }}
            priority
          />
          <Typography
            variant="h5"
            title={'PocketShield'}
            sx={{
              mt: 0.5,
              fontFamily: '"Plus Jakarta Sans", sans-serif',
              fontWeight: 600,
              color: 'rgb(85,152,229)',
            }}
          >
            {'PocketShield'}
          </Typography>
        </div>
      </header>

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '16px' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            aria-label="token classification tabs"
            sx={{
              '& .MuiTab-root': {
                textTransform: 'none',
                fontWeight: 500,
                fontSize: '1rem',
                color: '#5F6368',
                minWidth: 100,
                padding: '12px 16px',
                '&.Mui-selected': { color: '#1a73e8', fontWeight: 500 },
              },
              '& .MuiTabs-indicator': { backgroundColor: '#1a73e8' },
            }}
          >
            <Tab label="Reports Dashboard" value="jobs" />
            <Tab label="Usage Stats" value="monitoring" />
          </Tabs>
        </Box>

        {/* Tab Content Sections */}
        {/* Using conditional rendering for clarity; display:none also works */}
        {tabValue === 'monitoring' && (
          <Dashboard /> // Assuming Dashboard might also use data like trainReport
        )}
        {tabValue === 'jobs' && <Jobs />}

        {/* Example of how you might show loading/error for the report if it's tied to a tab */}
        {tabValue === 'monitoring' && isLoadingReport && <CircularProgress />}
        {tabValue === 'monitoring' && reportError && (
          <Typography color="error">{reportError}</Typography>
        )}
      </main>
    </div>
  );
}

// --- The default export Page component ---
export default function Page() {
  // This component now just sets up the Suspense boundary.
  // The fallback can be a simple loading message or a more sophisticated skeleton UI.
  return (
    <Suspense
      fallback={
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
          <CircularProgress />
          <Typography variant="h6" component="p" sx={{ ml: 2 }}>
            Loading page...
          </Typography>
        </Box>
      }
    >
      <PageContents />
    </Suspense>
  );
}
