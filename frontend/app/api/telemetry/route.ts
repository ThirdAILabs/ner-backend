import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';

// PostgreSQL connection configuration for telemetry
const pool = new Pool({
  host: process.env.POSTGRES_HOST || 'pocketllm-shield.cjhmqzlwgr5q.us-east-1.rds.amazonaws.com',
  port: parseInt(process.env.POSTGRES_PORT || '5432'),
  database: process.env.POSTGRES_DATABASE || 'postgres',
  user: process.env.POSTGRES_USER || 'telemetry_writer',
  password: process.env.POSTGRES_PASSWORD!,
  ssl: {
    rejectUnauthorized: false, // For AWS RDS
  },
  max: 5, // Maximum number of clients in the pool
  idleTimeoutMillis: 30000, // Close idle clients after 30 seconds
  connectionTimeoutMillis: 10000, // Return an error after 10 seconds if connection could not be established
});

// Handle pool errors
pool.on('error', (err) => {
  console.error('Unexpected error on idle PostgreSQL client', err);
});

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { username, timestamp, user_machine, event } = body;

    // Validate required fields
    if (!username || !timestamp || !user_machine || !event) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    const client = await pool.connect();
    try {
      const query = `
        INSERT INTO telemetry_events (username, timestamp, user_machine, event)
        VALUES ($1, $2, $3, $4)
      `;
      
      await client.query(query, [
        username,
        timestamp,
        user_machine,
        JSON.stringify(event)
      ]);

      return NextResponse.json({ success: true }, { status: 201 });
    } catch (error) {
      console.error('Error inserting telemetry event:', error);
      return NextResponse.json(
        { error: 'Failed to insert telemetry event' },
        { status: 500 }
      );
    } finally {
      client.release();
    }
  } catch (error) {
    console.error('Error processing telemetry request:', error);
    return NextResponse.json(
      { error: 'Invalid request body' },
      { status: 400 }
    );
  }
} 