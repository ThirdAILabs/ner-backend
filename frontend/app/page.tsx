'use client';

import * as _ from 'lodash';
import Image from 'next/image';
import Link from 'next/link';
import { useEffect } from 'react';
import { Box, Typography } from '@mui/material';
import { Card, CardTitle } from '@/components/ui/card/index';
import { FileText, MessageSquare } from 'lucide-react';
import useTelemetry from '@/hooks/useTelemetry';

function Choice({
  title,
  subtitle,
  icon,
  href,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  href: string;
}) {
  return (
    <Link href={href}>
      <Card className="w-[300px] h-[250px] flex justify-center items-center hover:scale-105 transition-transform duration-200 hover:shadow-lg cursor-pointer">
        <Box>
          {/* {icon}
          <CardTitle className="text-center text-gray-500">{title}</CardTitle> */}
          <div className="flex justify-center mt-4">{icon}</div>
          <CardTitle className="text-center text-gray-500">{title}</CardTitle>
          <Typography variant="subtitle2" className="text-center text-gray-400 pt-3">
            {subtitle}
          </Typography>
        </Box>
      </Card>
    </Link>
  );
}

export default function Page() {
  const recordEvent = useTelemetry();

  // Record initial page load
  useEffect(() => {
    recordEvent({
      UserAction: 'view',
      UIComponent: 'Page Load',
      UI: 'Home Page'
    });
  }, []);

  return (
    <div style={{ width: '75%', minHeight: 'calc(100vh-20px)', margin: '0 auto' }}>
      <header style={{ width: '100%', padding: '16px', borderBottom: '1px solid #e0e0e0' }}>
        <div
          style={{
            maxWidth: '1200px',
            margin: '0 auto',
            marginBottom: '8px',
            display: 'flex',
            flexDirection: 'row',
            gap: '20px',
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

      <main
        style={{
          maxWidth: '1200px',
          height: '80%',
          margin: '0 auto',
          padding: '50px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Box sx={{ position: 'relative' }}>
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography
              className="text-gray-500"
              variant="h5"
              title={'PocketShield'}
              sx={{
                mt: 0.5,
                fontFamily: '"Plus Jakarta Sans", sans-serif',
                fontWeight: 600,
              }}
            >
              What would you like to do today?
            </Typography>
          </Box>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '50px',
              width: '100%',
            }}
          >
            <Choice
              title="Scan Files"
              subtitle="Completely Airgapped"
              icon={
                <FileText
                  className="text-[rgb(85,152,229)]"
                  size={80}
                  style={{ marginBottom: 24 }}
                  strokeWidth={1}
                />
              }
              href="/token-classification/landing"
            />
            <Choice
              title="SafeGPT"
              subtitle="With Local LLM Guardrails"
              icon={
                <MessageSquare
                  className="text-[rgb(85,152,229)]"
                  size={80}
                  style={{ marginBottom: 24 }}
                  strokeWidth={1}
                />
              }
              href="/safegpt?id=new"
            />
          </Box>
        </Box>
      </main>
    </div>
  );
}