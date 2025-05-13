const http = require('http');

// Create a very simple server on port 3099
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    message: 'Server is running',
    url: req.url
  }));
});

server.listen(3099, () => {
  console.log('Server running at http://localhost:3099/');
}); 