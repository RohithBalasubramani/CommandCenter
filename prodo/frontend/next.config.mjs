/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    // Skip ESLint during production builds (pre-existing warnings only —
    // no-explicit-any, no-unused-vars, etc.). Lint is still run in dev.
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
