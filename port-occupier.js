const http = require('http');

// Create a server that will occupy port 8000
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('Port 8000 is occupied by this server\n');
});

// Listen on port 8000
server.listen(8000, () => {
  console.log('TEST SERVER RUNNING - PORT 8000 IS NOW OCCUPIED');
  console.log('Press Ctrl+C to stop');
});

// Handle graceful shutdown
process.on('SIGINT', function() {
  console.log('Stopping test server...');
  server.close(() => {
    console.log('Test server on port 8000 stopped');
    process.exit(0);
  });
}); 