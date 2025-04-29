import * as React from 'react';
import Box from '@mui/joy/Box';
import Radio from '@mui/joy/Radio';
import RadioGroup from '@mui/joy/RadioGroup';
import Typography from '@mui/joy/Typography';

export default function ExampleSegmentedControls({ tabs }: Tabs) {
    const [justify, setJustify] = React.useState(tabs[0]);
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <RadioGroup
                orientation="horizontal"
                aria-labelledby="segmented-controls-example"
                name="justify"
                value={justify}
                onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                    setJustify(event.target.value)
                }
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
