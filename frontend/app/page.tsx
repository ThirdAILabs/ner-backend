'use client';

import * as _ from 'lodash';
import Image from 'next/image';
import Link from 'next/link';
import { Box, Typography } from '@mui/material';
import { FileText, MessageSquare } from 'lucide-react';
import CustomisableCard from '@/components/ui/cards/customisableCard';

const BACKGROUND_IMAGE_URL = {
  Scan: '/scan.svg',
  SafeGPT: '/safegpt.svg',
};

function Choice({
  title,
  subtitle,
  icon,
  href,
  backgroundImage,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
  href: string;
  backgroundImage: string;
}) {
  return (
    <Link href={href}>
      <CustomisableCard
        children={
          <div className="flex flex-col h-full ml-12 mt-16">
            <span className="text-[#5598E5] text-2xl font-semibold">{title}</span>
            <span className="mt-2 text-xs text-gray-500 w-56">{subtitle}</span>
          </div>
        }
        width="26vw"
        height="26vw"
        backgroundImage={backgroundImage}
      />
    </Link>
  );
}
export default function Page() {
  console.log('Page loaded', BACKGROUND_IMAGE_URL);

  return (
    <>
      <div
        style={{
          width: '100%',
          height: '100vh',
          position: 'fixed',
          top: 0,
          left: 0,
          margin: '0',
          padding: '0',
          background: 'linear-gradient(135deg, #74AFF4 5%, #5598E5 33%, #2F547F 100%)',
          zIndex: -1,
        }}
      />

      <div
        style={{
          width: '100%',
          height: '100vh',
          position: 'relative',
        }}
      >
        <header
          style={{
            width: '100%',
            padding: '16px',
          }}
        >
          <div
            style={{
              margin: '0 auto',
              marginBottom: '8px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '20px',
              justifyContent: 'center',
              paddingTop: '60px',
            }}
          >
            <Image
              src="/thirdaiWhite.svg"
              alt="ThirdAI Logo"
              width={40}
              height={40}
              style={{ objectFit: 'contain', color: 'white' }}
              priority
            />
            <Typography
              variant="h5"
              title={'PocketShield'}
              sx={{
                mt: 0.5,
                fontFamily: '"Plus Jakarta Sans", sans-serif',
                fontWeight: 600,
                color: 'white',
              }}
            >
              {'Welcome to PocketShield'}
            </Typography>
          </div>
        </header>

        <main
          style={{
            height: '76.2%',
            marginTop: '0.8%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'white',
            marginInline: '4px',
            borderRadius: '16px',
          }}
        >
          <Box sx={{ position: 'relative', marginTop: '-100px' }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
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
                subtitle="Scan your files for sensitive data, completely air-gapped"
                icon={
                  <FileText
                    className="text-[rgb(85,152,229)]"
                    size={80}
                    style={{ marginBottom: 24 }}
                    strokeWidth={1}
                  />
                }
                href="/token-classification/landing"
                backgroundImage={BACKGROUND_IMAGE_URL.Scan}
              />
              <Choice
                title="SafeGPT"
                subtitle="The ChatGPT you know and love, with local LLM  guardrails"
                icon={
                  <MessageSquare
                    className="text-[rgb(85,152,229)]"
                    size={80}
                    style={{ marginBottom: 24 }}
                    strokeWidth={1}
                  />
                }
                href="/safegpt?id=new"
                backgroundImage={BACKGROUND_IMAGE_URL.SafeGPT}
              />
            </Box>
          </Box>
        </main>
      </div>
    </>
  );
}
