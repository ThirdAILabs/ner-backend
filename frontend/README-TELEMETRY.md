# PostgreSQL Telemetry Configuration

This application has been updated to use PostgreSQL instead of Supabase for telemetry data collection.

## Environment Variables

Create a `.env.local` file in the frontend directory with the following variables:

```bash
# PostgreSQL Configuration for Telemetry
# Using telemetry_writer user (write-only permissions)
POSTGRES_HOST=pocketllm-shield.cjhmqzlwgr5q.us-east-1.rds.amazonaws.com
POSTGRES_PORT=5432
POSTGRES_DATABASE=postgres
POSTGRES_USER=telemetry_writer
POSTGRES_PASSWORD=YOUR_TELEMETRY_WRITER_PASSWORD_HERE
```

## Database Schema

The telemetry data is stored in the `telemetry_events` table:

```sql
Table "public.telemetry_events"
    Column    |           Type           | Nullable |                   Default                    
--------------+--------------------------+----------+----------------------------------------------
 id           | integer                  | not null | nextval('telemetry_events_id_seq'::regclass)
 username     | text                     | not null | 
 timestamp    | timestamp with time zone | not null | 
 user_machine | text                     | not null | 
 event        | jsonb                    | not null | 
```

## Security

- Uses the `telemetry_writer` user which has INSERT permissions only
- No read permissions for security
- SSL connection required
- Database credentials are server-side only (not exposed to client)

## API Endpoint

Telemetry data is sent via POST request to `/api/telemetry` with the following payload:

```typescript
{
  username: string;     // User ID from localStorage or "anonymous"
  timestamp: string;    // ISO timestamp
  user_machine: string; // Browser user agent
  event: {              // Telemetry event data
    UserAction: string;   // e.g., "click", "view", "select"
    UIComponent: string;  // e.g., "Usage Stats Tab"
    UI: string;          // e.g., "Token Classification Page"
    data?: any;          // Optional additional data
  };
}
```

## Usage

The telemetry system automatically tracks:

1. **Page views** - When users navigate to different sections
2. **User interactions** - Clicks, selections, form submissions
3. **Component usage** - Which UI components are being used

No changes needed to existing telemetry tracking code - the `useTelemetry()` hook remains the same.

## Testing Connection

To test the PostgreSQL connection, you can use:

```bash
psql 'postgresql://telemetry_writer:PASSWORD@pocketllm-shield.cjhmqzlwgr5q.us-east-1.rds.amazonaws.com:5432/postgres?sslmode=require'
```

Replace `PASSWORD` with the actual `telemetry_writer` password. 