import JobDetailClient from './client';

// With static export removed, we no longer need generateStaticParams
// Dynamic routes will be handled at runtime

export default function JobDetailPage({ params }: { params: { deploymentId: string, jobId: string } }) {
  return <JobDetailClient deploymentId={params.deploymentId} jobId={params.jobId} />;
} 