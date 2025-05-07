/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
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
    unoptimized: true, // Keep for Electron compatibility
  },
  // output: 'export', // Removed static export configuration
  distDir: 'out', // Output directory for build files
  trailingSlash: true, // Add trailing slashes to URLs for better file serving
  webpack(config) {
    config.module.rules.push({
      test: /\.svg$/i,
      issuer: /\.[jt]sx?$/,
      use: ['@svgr/webpack'],
    });

    return config;
  },
  // Note: rewrites won't work with static export
  // In Electron, we'll handle this in the main process
  // or use absolute URLs in the axios config
};

module.exports = nextConfig;
