import * as React from 'react';
import { emphasize, styled } from '@mui/material/styles';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Chip from '@mui/material/Chip';
import HomeIcon from '@mui/icons-material/Home';

const StyledBreadcrumb = styled(Chip)(({ theme }) => {
    return {
        backgroundColor: theme.palette.grey[100],
        height: theme.spacing(3),
        color: theme.palette.text.primary,
        fontWeight: theme.typography.fontWeightRegular,
        userSelect: 'none',
        cursor: 'pointer',
        '&:hover, &:focus': {
            backgroundColor: emphasize(theme.palette.grey[100], 0.06),
            ...theme.applyStyles('dark', {
                backgroundColor: emphasize(theme.palette.grey[800], 0.06),
            }),
        },
        '&:active': {
            boxShadow: theme.shadows[1],
            backgroundColor: emphasize(theme.palette.grey[100], 0.12),
            ...theme.applyStyles('dark', {
                backgroundColor: emphasize(theme.palette.grey[800], 0.12),
            }),
        },
        ...theme.applyStyles('dark', {
            backgroundColor: theme.palette.grey[800],
        }),
    };
}) as typeof Chip; // TypeScript only: need a type cast here because https://github.com/Microsoft/TypeScript/issues/26591

function handleClick(event: React.MouseEvent<Element, MouseEvent>) {
    event.preventDefault();
    console.info('You clicked a breadcrumb.');
}

interface CustomizedBreadcrumbsProps {
    breadcrumbs: BreadCrumb[];
}

export default function CustomizedBreadcrumbs({ breadcrumbs }: CustomizedBreadcrumbsProps) {
    return (
        <div role="presentation" onClick={handleClick}>
            <Breadcrumbs aria-label="breadcrumb">
                {
                    breadcrumbs.map((breadcrumb) => {
                        return (
                            <StyledBreadcrumb
                                component="a"
                                href="#"
                                label="Home"
                                icon={<HomeIcon fontSize="small" />}
                            />
                        )
                    })
                }
            </Breadcrumbs>
        </div>
    );
}
