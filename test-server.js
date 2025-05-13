const http = require('http');

// Create a simple HTTP server
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('Test server running on port 8000\n');
});

// Listen on port 8000
server.listen(8000, () => {
  console.log('Test server is running on port 8000');
});

console.log('Press Ctrl+C to stop the server'); 