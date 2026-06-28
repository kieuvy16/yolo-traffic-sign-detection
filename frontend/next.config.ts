/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable CORS for API requests to Flask backend
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/:path*',
      },
    ];
  },
  // Allow images from data URLs and external sources
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '5000',
        pathname: '/**',
      },
    ],
    // Allow data URLs for base64 images
    dangerouslyAllowSVG: true,
  },
  // Disable strict mode for development
  reactStrictMode: true,
};

export default nextConfig;
