import TokenClassificationClient from './client';

// This function is needed for static site generation with dynamic routes
export function generateStaticParams() {
  // Pre-generate a default deploymentId for static export
  // In a real app, you would fetch all possible IDs from your API
  return [
    { deploymentId: 'default' },
    { deploymentId: 'PII' },
    { deploymentId: 'Home' }
  ];
}

export default function Page({ params }: { params: { deploymentId: string } }) {
  return <TokenClassificationClient deploymentId={params.deploymentId} />;
} 