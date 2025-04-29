import * as React from 'react';
import Box from '@mui/joy/Box';
import Radio from '@mui/joy/Radio';
import RadioGroup from '@mui/joy/RadioGroup';

export default function ExampleSegmentedControls({ tabs, onChange, value }: Tabs) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
      <RadioGroup
        orientation="horizontal"
        aria-labelledby="segmented-controls-example"
        name="justify"
        value={value}
        onChange={(event: React.ChangeEvent<HTMLInputElement>) => onChange?.(event.target.value)}
        sx={{
          minHeight: 32,
          padding: '4px',
          borderRadius: '8px',
          bgcolor: 'neutral.softBg',
          '--RadioGroup-gap': '4px',
          '--Radio-actionRadius': '4px',
        }}
      >
        {tabs.map((item) => (
          <Radio
            key={item}
            color="neutral"
            value={item}
            disableIcon
            label={item}
            variant="plain"
            sx={{ px: 6, alignItems: 'center', userSelect: 'none' }}
            slotProps={{
              action: ({ checked, focusVisible }) => ({
                sx: {
                  ...(checked && {
                    bgcolor: 'background.surface',
                    boxShadow: 'sm',
                    '&:hover': {
                      bgcolor: 'background.surface',
                    },
                  }),
                },
              }),
            }}
          />
        ))}
      </RadioGroup>
    </Box>
  );
}
