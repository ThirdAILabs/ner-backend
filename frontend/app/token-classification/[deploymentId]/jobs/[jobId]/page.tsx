import JobDetailClient from './client';

// This function is needed for static site generation with dynamic routes
export function generateStaticParams() {
  // Pre-generate for the deploymentIds we know about
  return [
    { deploymentId: 'default', jobId: 'default' },
    { deploymentId: 'PII', jobId: 'default' },
    { deploymentId: 'Home', jobId: 'default' },
    { deploymentId: 'Home', jobId: '1065a99d-8234-4ba6-9e63-2a7d57dcd29a' },
    { deploymentId: 'Home', jobId: '4c206909-e943-4dde-9c0b-8d91d82fe08a' },
    { deploymentId: 'Home', jobId: '93ba2a2d-78e3-497a-8ad0-1e5fb64c1f2a' }
  ];
}

export default function JobDetailPage({ params }: { params: { deploymentId: string, jobId: string } }) {
  return <JobDetailClient deploymentId={params.deploymentId} jobId={params.jobId} />;
} 