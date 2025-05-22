'use client';

import { Box, Typography } from '@mui/material'; // Added CircularProgress
import * as _ from 'lodash';
import Image from 'next/image';
import { Card, CardTitle } from '@/components/ui/card/index';
import { FileText, MessageSquare } from 'lucide-react';
import Link from 'next/link';

function Choice({ title, icon, href }: { title: string; icon: React.ReactNode; href: string }) {
  return (
    <Link href={href}>
      <Card className="w-[300px] h-[250px] flex justify-center items-center hover:scale-105 transition-transform duration-200 hover:shadow-lg cursor-pointer">
        <Box>
          {icon}
          <CardTitle className="text-center">{title}</CardTitle>
        </Box>
      </Card>
    </Link>
  );
}

export default function Page() {
  return (
    <div style={{ width: '75%', minHeight: '100vh', margin: '0 auto' }}>
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
              variant="h5"
              title={'PocketShield'}
              sx={{
                mt: 0.5,
                fontFamily: '"Plus Jakarta Sans", sans-serif',
                fontWeight: 600,
                color: '#000',
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
              icon={
                <FileText size={80} color="#000" style={{ marginBottom: 24 }} strokeWidth={1} />
              }
              href="/token-classification/landing"
            />
            <Choice
              title="SafeGPT"
              icon={
                <MessageSquare
                  size={80}
                  color="#000"
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
