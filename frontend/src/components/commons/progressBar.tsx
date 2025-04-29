import * as React from 'react';
import LinearProgress, { LinearProgressProps } from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';

function LinearProgressWithLabel(
  props: LinearProgressProps & { value: number; displaytext: string }
) {
  const { displaytext } = props;
  return (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Box sx={{ width: '80%', mr: 1 }}>
        <LinearProgress variant="determinate" {...props} />
      </Box>
      <Box sx={{ minWidth: 35 }}>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          {displaytext}
        </Typography>
      </Box>
    </Box>
  );
}

export default function LinearProgressBar({
  value,
  displaytext,
}: {
  value: number;
  displaytext: string;
}) {
  return (
    <Box sx={{ width: '80%' }}>
      <LinearProgressWithLabel value={value} displaytext={displaytext} />
    </Box>
  );
}
