import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';

// Force Node.js runtime so we can use the 'pg' module
export const runtime = 'nodejs';

// Initialize a connection pool for AWS RDS Postgres
const pool = new Pool({
  host: process.env.PGHOST,
  port: Number(process.env.PGPORT || 5432),
  user: process.env.PGUSER,
  password: process.env.PGPASSWORD,
  database: process.env.PGDATABASE,
  ssl: { rejectUnauthorized: false },
});

export async function POST(request: NextRequest) {
  try {
    const { username, timestamp, user_machine, event } = await request.json();
    await pool.query(
      'INSERT INTO public.telemetry_events (username, timestamp, user_machine, event) VALUES ($1, $2, $3, $4)',
      [username, timestamp, user_machine, event]
    );
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Telemetry insert error:', error);
    return NextResponse.json({ error: 'Database insert error' }, { status: 500 });
  }
} 