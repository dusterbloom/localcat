/** @type {import('next').NextConfig} */
const isProduction = process.env.NODE_ENV === 'production';
const nextConfig = {
  ...(isProduction && { output: 'export' }),
  images: { unoptimized: true },
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
  async rewrites() {
    if (!isProduction) {
      return [
        {
          source: '/api/:path*',
          destination: 'http://0.0.0.0:7860/api/:path*',
        },
      ];
    }
    return [];
  },
};

module.exports = nextConfig;
