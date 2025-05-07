'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useDeploymentIds } from '../lib/electronRouter';

export default function Home() {
  const { deploymentIds, isLoading, error } = useDeploymentIds();

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-2xl font-bold mb-4">Pocket Shield</h1>
      <p className="mb-4">Welcome to the simplified platform</p>
      
      {isLoading ? (
        <p>Loading available deployments...</p>
      ) : error ? (
        <div className="text-red-500 mb-4">{error}</div>
      ) : (
        <div className="flex flex-col gap-4 w-full max-w-md">
          <h2 className="text-xl font-semibold">Available Deployments:</h2>
          {deploymentIds.map(deploymentId => (
            <Link 
              key={deploymentId}
              href={`/token-classification/${deploymentId}`}
              className="px-4 py-3 bg-blue-500 text-white rounded hover:bg-blue-600 text-center"
            >
              {deploymentId}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
} 