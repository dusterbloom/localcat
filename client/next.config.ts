import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Only use static export for production builds (npm run build)
  // Dev mode needs dynamic server for CSS/HMR
  ...(process.env.NODE_ENV === 'production' && { output: 'export' }),
  // Note: rewrites don't work with static export
  // API calls should use NEXT_PUBLIC_SERVER_URL environment variable
};

export default nextConfig;
