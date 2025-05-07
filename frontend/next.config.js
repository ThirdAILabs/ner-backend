/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',  // Enable static export
  distDir: 'out',    // Set output directory
  images: {
    unoptimized: true,  // Required for static export
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
  // Remove rewrites when in export mode since they don't work with static export
  // We'll handle API calls directly in Electron
};

module.exports = nextConfig;
