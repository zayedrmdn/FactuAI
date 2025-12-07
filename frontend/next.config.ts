import type { NextConfig } from 'next';
import withBundleAnalyzer from '@next/bundle-analyzer';

// Initialize the analyzer wrapper
const withAnalyzer = withBundleAnalyzer({
  enabled: process.env.ANALYZE === 'true',
});

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/api/:path*',
      },
    ];
  },
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '5000',
        pathname: '/**',
      },
    ],
  },
};

// Wrap the config with the analyzer
export default withAnalyzer(nextConfig);
