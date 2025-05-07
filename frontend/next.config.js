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
    unoptimized: true, // For static export
  },
  output: 'export', // Enable static exports for Electron
  distDir: 'out', // Output directory for static files
  trailingSlash: true, // Add trailing slashes to URLs for better static file serving
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
