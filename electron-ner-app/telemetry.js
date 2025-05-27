import { Client } from 'pg';

// PostgreSQL client for telemetry
let pgClient = null;

// Initialize PostgreSQL connection
export async function initTelemetry() {
  try {
    pgClient = new Client({
      host: process.env.POSTGRES_HOST || 'pocketllm-shield.cjhmqzlwgr5q.us-east-1.rds.amazonaws.com',
      port: parseInt(process.env.POSTGRES_PORT || '5432'),
      database: process.env.POSTGRES_DATABASE || 'postgres',
      user: process.env.POSTGRES_USER || 'telemetry_writer',
      password: process.env.POSTGRES_PASSWORD || 'kLKDPmv21w93oam!93',
      ssl: {
        rejectUnauthorized: false, // For AWS RDS
      },
    });

    await pgClient.connect();
    console.log('Telemetry PostgreSQL connected successfully');
  } catch (error) {
    console.error('Failed to connect to telemetry database:', error);
    pgClient = null;
  }
}

// Insert telemetry event
export async function insertTelemetryEvent(data) {
  if (!pgClient) {
    console.warn('Telemetry database not connected');
    return;
  }

  try {
    const query = `
      INSERT INTO telemetry_events (username, timestamp, user_machine, event)
      VALUES ($1, $2, $3, $4)
    `;
    
    await pgClient.query(query, [
      data.username,
      data.timestamp,
      data.user_machine,
      JSON.stringify(data.event)
    ]);
  } catch (error) {
    console.error('Error inserting telemetry event:', error);
  }
}

// Close telemetry connection
export async function closeTelemetry() {
  if (pgClient) {
    try {
      await pgClient.end();
      console.log('Telemetry database connection closed');
    } catch (error) {
      console.error('Error closing telemetry connection:', error);
    }
  }
}

// ES modules exports are handled by individual export statements above 