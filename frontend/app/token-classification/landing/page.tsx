'use client';

import { Suspense } from 'react';
import { useEffect, useState } from 'react';
import { Tabs, Tab, Box, CircularProgress, Typography } from '@mui/material';
import * as _ from 'lodash';
import { useSearchParams } from 'next/navigation';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import Dashboard from '../dashboard';
import Jobs from '../jobs';
import useTelemetry from '@/hooks/useTelemetry';
import ModelCustomization from './ModelCustomization';
import { useLicense } from '@/hooks/useLicense';

function PageContents() {
  const { isEnterprise } = useLicense();

  const searchParams = useSearchParams();
  const defaultTab = searchParams.get('tab') || 'jobs';
  const [tabValue, setTabValue] = useState(defaultTab);
  const recordEvent = useTelemetry();

  // Record initial page load
  useEffect(() => {
    recordEvent({
      UserAction: 'View Report Dashboard',
      UIComponent: 'Report Dashboard',
      Page: 'Report Dashboard Page',
    });
  }, []);

  // Update tabValue if searchParams change after initial load (e.g., browser back/forward)
  useEffect(() => {
    setTabValue(searchParams.get('tab') || 'jobs');
  }, [searchParams]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: string) => {
    setTabValue(newValue);
    if (newValue === 'monitoring') {
      recordEvent({
        UserAction: 'Click on Usage Stats Tab',
        UIComponent: 'Usage Stats Tab',
        Page: 'Report Dashboard Page',
      });
    } else if (newValue === 'jobs') {
      recordEvent({
        UserAction: 'Click on Report Dashboard Tab',
        UIComponent: 'Reports Dashboard Tab',
        Page: 'Report Dashboard Page',
      });
    }
  };

  return (
    // 30px is the height of the title bar
    <div
      style={{
        width: '90%',
        minHeight: 'calc(100vh - 30px)',
        margin: '0 auto',
      }}
    >
      <header
        style={{
          width: '100%',
          padding: '16px',
          borderBottom: '1px solid #e0e0e0',
          display: 'flex',
          alignItems: 'center',
          position: 'relative',
        }}
      >
        {/* Left - Back Button */}
        <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-start' }}>
          <Button variant="outline" size="sm" asChild>
            <Link href={`/`} className="flex items-center">
              <ArrowLeft className="mr-1 h-4 w-4" /> Back
            </Link>
          </Button>
        </div>

        {/* Center - Logo and Title */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '20px',
            justifyContent: 'center',
            margin: '0 auto',
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
            title="PocketShield"
            sx={{
              fontFamily: '"Plus Jakarta Sans", sans-serif',
              fontWeight: 600,
              color: 'rgb(85,152,229)',
            }}
          >
            PocketShield
          </Typography>
        </div>

        {/* Right - Empty to balance */}
        <div style={{ flex: 1 }} />
      </header>

      <main style={{ margin: '0 auto', padding: '16px' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            aria-label="scans dashboard tabs"
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
            <Tab label="Scans Dashboard" value="jobs" />
            <Tab label="Usage Stats" value="monitoring" />
            {isEnterprise && <Tab label="Model Customization" value="customization" />}
          </Tabs>
        </Box>

        {tabValue === 'monitoring' && <Dashboard />}
        {tabValue === 'jobs' && <Jobs />}
        {isEnterprise && tabValue === 'customization' && <ModelCustomization />}
      </main>
    </div>
  );
}

export default function Page() {
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
