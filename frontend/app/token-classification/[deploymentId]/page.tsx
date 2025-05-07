import TokenClassificationClient from './client';

// With static export removed, we no longer need generateStaticParams
// Dynamic routes will be handled at runtime

export default function Page({ params }: { params: { deploymentId: string } }) {
  return <TokenClassificationClient deploymentId={params.deploymentId} />;
} 