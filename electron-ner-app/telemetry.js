import { Client } from 'pg';

// PostgreSQL client for telemetry
let pgClient = null;
let isConnected = false;

// Initialize PostgreSQL connection (silent failure)
export async function initTelemetry() {
  try {
    pgClient = new Client({
      host: process.env.POSTGRES_HOST || 'pocketllm-shield.cjhmqzlwgr5q.us-east-1.rds.amazonaws.com',
      port: parseInt(process.env.POSTGRES_PORT || '5432'),
      database: process.env.POSTGRES_DATABASE || 'postgres',
      user: process.env.POSTGRES_USER || 'telemetry_writer',
      password: process.env.POSTGRES_PASSWORD || 'kLKDPmv21w93oam!93',
      ssl: {
        rejectUnauthorized: false,
      },
      connectionTimeoutMillis: 5000, // Shorter timeout for offline scenarios
      query_timeout: 3000,
    });

    // Add error handlers to prevent uncaught exceptions
    pgClient.on('error', (err) => {
      console.log('Telemetry database error (silent):', err.message);
      isConnected = false;
      pgClient = null;
    });

    // Handle connection termination
    pgClient.on('end', () => {
      console.log('Telemetry database connection ended');
      isConnected = false;
      pgClient = null;
    });

    // Handle unexpected connection termination
    pgClient.connection?.socket?.on('error', (err) => {
      console.log('Telemetry socket error (silent):', err.message);
      isConnected = false;
      pgClient = null;
    });

    // Handle connection close
    pgClient.connection?.socket?.on('close', () => {
      console.log('Telemetry socket closed unexpectedly');
      isConnected = false;
      pgClient = null;
    });

    await pgClient.connect();
    isConnected = true;
    console.log('Telemetry PostgreSQL connected successfully');
  } catch (error) {
    // Silent failure - just log to console
    console.log('Telemetry unavailable (offline or connection failed):', error.message);
    pgClient = null;
    isConnected = false;
  }
}

// Insert telemetry event (silent failure)
export async function insertTelemetryEvent(data) {
  // Quick check if we're not connected
  if (!pgClient || !isConnected) {
    console.log('Telemetry event skipped (no connection)');
    return;
  }

  try {
    const query = `
      INSERT INTO telemetry_events (username, timestamp, pocketshield_version, user_machine, event)
      VALUES ($1, $2, $3, $4, $5)
    `;
    
    await pgClient.query(query, [
      data.username,
      data.timestamp,
      data.pocketshield_version,
      data.user_machine,
      JSON.stringify(data.event)
    ]);
    
    console.log('Telemetry event recorded');
  } catch (error) {
    // Silent failure - just log to console
    console.log('Telemetry event failed (silent):', error.message);
    
    // Mark as disconnected if it's a connection error
    if (error.code === 'ECONNRESET' || error.code === 'ETIMEDOUT' || error.code === 'ENOTFOUND') {
      isConnected = false;
      pgClient = null;
    }
  }
}

// Close telemetry connection (silent)
export async function closeTelemetry() {
  if (pgClient) {
    try {
      await pgClient.end();
      console.log('Telemetry database connection closed');
    } catch (error) {
      console.log('Error closing telemetry connection (silent):', error.message);
    } finally {
      pgClient = null;
      isConnected = false;
    }
  }
}

// Optional: Add a function to check if telemetry is available
export function isTelemetryAvailable() {
  return isConnected && pgClient !== null;
}

// Optional: Retry connection function (call this periodically if needed)
export async function retryTelemetryConnection() {
  if (!isConnected) {
    console.log('Attempting to reconnect telemetry...');
    await initTelemetry();
  }
}