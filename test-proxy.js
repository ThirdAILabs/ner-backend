const express = require('express');

// Create a simple test backend on port 8001
const app = express();

app.get('/', (req, res) => {
  res.send('Test proxy server is running');
});

// Add API endpoint to simulate backend
app.get('/api/v1/health', (req, res) => {
  res.json({ status: 'ok', message: 'API is healthy' });
});

app.get('/api/v1/reports', (req, res) => {
  res.json([
    { id: 1, name: 'Test Report 1' },
    { id: 2, name: 'Test Report 2' }
  ]);
});

// Enable CORS
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, PUT, POST, DELETE');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Start server on port 3099 - same as our proxy would use
const PORT = 3099;
app.listen(PORT, () => {
  console.log(`Test server running on http://localhost:${PORT}`);
  console.log('Try these URLs:');
  console.log(`- http://localhost:${PORT}/`);
  console.log(`- http://localhost:${PORT}/api/v1/health`);
  console.log(`- http://localhost:${PORT}/api/v1/reports`);
}); 