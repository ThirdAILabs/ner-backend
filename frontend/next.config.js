/** @type {import('next').NextConfig} */
const nextConfig = {
  // PostgreSQL configuration is now handled server-side via API routes
  // No need to expose credentials to the client
  // Conditionally use static export only for Electron builds
  ...(process.env.BUILD_STATIC === 'true' 
    ? { output: 'export', distDir: 'out' } 
    : {}),
  trailingSlash: true,
  images: {
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'avatars.githubusercontent.com',
      },
      {
        protocol: 'https',
        hostname: '*.public.blob.vercel-storage.com',
      },
    ],
  },
  webpack(config) {
    config.module.rules.push({
      test: /\.svg$/i,
      issuer: /\.[jt]sx?$/,
      use: ['@svgr/webpack'],
    });

    return config;
  },
};

module.exports = nextConfig;
