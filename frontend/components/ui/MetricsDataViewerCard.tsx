import { Card, CardContent, Box, Typography } from '@mui/material';

interface MetricsDataViewerCardProps {
  value: number | string;
  label: string;
}

function getFontSize(value: string) {
  if (!value) return '2rem';
  if (value.length > 7) return '1.25rem';
  if (value.length > 5) return '1.5rem';
  if (value.length > 3) return '1.75rem';

  return '2rem';
}

const MetricsDataViewerCard: React.FC<MetricsDataViewerCardProps> = ({ value, label }) => {
  return (
    <Card
      sx={{
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        bgcolor: 'white',
        borderRadius: '12px',
        border: '1px solid #e5e7eb',
        transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        },
      }}
    >
      <CardContent
        sx={{
          p: 3,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          '&:last-child': { pb: 3 },
        }}
      >
        <Box
          sx={{
            height: 100,
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
          }}
        >
          <Typography
            sx={{
              fontSize: getFontSize(String(value)),
              fontWeight: 600,
              color: '#4a5568',
              textAlign: 'center',
              lineHeight: 1.2,
              maxWidth: '100%',
              wordBreak: 'break-word',
            }}
          >
            {value}
          </Typography>
        </Box>
        <Typography
          sx={{
            textAlign: 'center',
            fontSize: '0.875rem',
            color: '#64748b',
            fontWeight: 500,
          }}
        >
          {label}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default MetricsDataViewerCard;
