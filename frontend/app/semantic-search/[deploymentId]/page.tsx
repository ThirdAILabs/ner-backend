import SemanticSearchClient from './client';

// This function is needed for static site generation with dynamic routes
export function generateStaticParams() {
  // Pre-generate a default deploymentId for static export
  return [
    { deploymentId: 'default' },
    { deploymentId: 'search' },
    { deploymentId: 'Home' }
  ];
}

export default function SemanticSearchPage({ params }: { params: { deploymentId: string } }) {
  return <SemanticSearchClient deploymentId={params.deploymentId} />;
} 