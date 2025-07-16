import { SelectChangeEvent, FormControl, Select, MenuItem } from '@mui/material';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';

interface FilterDropdownProps {
  label?: string;
  value: string | number;
  options: { value: string | number; label: string }[];
  onChange: (event: SelectChangeEvent<string | number>) => void;
}

const FilterDropdown = ({ label = '', value, options, onChange }: FilterDropdownProps) => {
  return (
    <div className={`flex items-center gap-3`}>
      {label && <span className="text-md font-semibold text-gray-500">{label}</span>}
      <FormControl size="small">
        <Select
          value={value}
          onChange={onChange}
          IconComponent={KeyboardArrowDownIcon}
          MenuProps={{
            PaperProps: {
              sx: {
                mt: 1,
                boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
                borderRadius: '8px',
                '& .MuiMenuItem-root': {
                  fontSize: '14px',
                  padding: '10px 16px',
                  '&:hover': {
                    backgroundColor: '#F9FAFB',
                  },
                  '&.Mui-selected': {
                    backgroundColor: '#F3F4F6',
                    '&:hover': {
                      backgroundColor: '#F3F4F6',
                    },
                  },
                },
              },
            },
          }}
          sx={{
            minWidth: '140px',
            height: '36px',
            '& .MuiSelect-select': {
              fontSize: '14px',
              fontWeight: 500,
              py: 1,
              px: 1.5,
              bgcolor: 'white',
              border: '1px solid #E5E7EB',
              borderRadius: '6px !important',
              color: '#6B7280',
              fontFamily: '"Plus Jakarta Sans", sans-serif',
              '&:focus': {
                borderRadius: '6px',
              },
            },
            '& .MuiOutlinedInput-notchedOutline': {
              border: 'none',
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              border: 'none',
            },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
              border: 'none',
            },
            '& .MuiSelect-icon': {
              color: '#6B7280',
              right: '8px',
            },
          }}
        >
          {options.map((option) => (
            <MenuItem
              key={option.value}
              value={option.value}
              sx={{
                fontSize: '14px',
                fontWeight: 400,
                color: '#374151',
                fontFamily: '"Plus Jakarta Sans", sans-serif',
              }}
            >
              {option.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </div>
  );
};

export default FilterDropdown;
