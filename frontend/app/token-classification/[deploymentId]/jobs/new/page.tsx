import NewJobPageClient from './client';

// This function is needed for static site generation with dynamic routes
export function generateStaticParams() {
  // Pre-generate for the deploymentIds we know about
  return [
    { deploymentId: 'default' },
    { deploymentId: 'PII' },
    { deploymentId: 'Home' }
  ];
}

export default function NewJobPage({ params }: { params: { deploymentId: string } }) {
  return <NewJobPageClient deploymentId={params.deploymentId} />;
} 