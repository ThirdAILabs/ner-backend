import * as React from 'react';
import LinearProgress, { LinearProgressProps } from '@mui/material/LinearProgress';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';

function LinearProgressWithLabel(props: LinearProgressProps & { value: number, left_value: string, right_value: string }) {

  return (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <Box sx={{ width: '80%', mr: 1 }}>
        <LinearProgress variant="determinate" {...props} />
      </Box>
      <Box sx={{ minWidth: 35 }}>
        <Typography
          variant="body2"
          sx={{ color: 'text.secondary' }}
        >{`${props.left_value}/${props.right_value}`}</Typography>
      </Box>
    </Box>
  );
}

export default function LinearProgressBar({ value, left_value, right_value }: { value: number, left_value: string, right_value: string }) {
  return (
    <Box sx={{ width: '80%' }}>
      <LinearProgressWithLabel value={value} left_value={left_value} right_value={right_value} />
    </Box>
  );
}