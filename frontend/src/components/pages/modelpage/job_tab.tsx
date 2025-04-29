import React, { useEffect, useState } from 'react';
import { styled } from '@mui/material/styles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell, { tableCellClasses } from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import '../../../styles/pages/modelpage/job_tab.scss';
import LinearProgressBar from '../../common/progressBar';

interface JobEntry {
  name: string;
  source: string;
  initiated: string;
  progress: {
    status: string;
    completed?: string;
    current?: number;
    total?: number;
  };
}

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: 'black',
    color: '#ffffff',
    fontWeight: 500,
  },
  [`&.${tableCellClasses.body}`]: {
    fontSize: 14,
  },
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  '&:nth-of-type(even)': {
    backgroundColor: '#f9f9f9',
  },
  '&:last-child td, &:last-child th': {
    border: 0,
  },
}));

const JobTab: React.FC<{ jobEntries: JobEntry[] }> = ({ jobEntries }) => {
  return (
    <div className="job-tab">
      <button className="new-job-button">New</button>
      <TableContainer component={Paper} sx={{ maxHeight: 600 }}>
        <Table stickyHeader sx={{ minWidth: 700 }} aria-label="customized table">
          <TableHead>
            <TableRow>
              <StyledTableCell>Name</StyledTableCell>
              <StyledTableCell>Source</StyledTableCell>
              <StyledTableCell>Initiated</StyledTableCell>
              <StyledTableCell>Progress</StyledTableCell>
              <StyledTableCell>Action</StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {jobEntries.map((job, index) => (
              <StyledTableRow key={index}>
                <StyledTableCell>{job.name}</StyledTableCell>
                <StyledTableCell>{job.source}</StyledTableCell>
                <StyledTableCell>{job.initiated}</StyledTableCell>
                {job.progress.status === 'Completed' ? (
                  <StyledTableCell>
                    <span className="badge badge-success">{job.progress.status}</span>
                  </StyledTableCell>
                ) : (
                  <StyledTableCell>
                    {
                      <LinearProgressBar
                        value={
                          job.progress.current && job.progress.total
                            ? Math.round((job.progress.current / job.progress.total) * 100)
                            : 0
                        }
                        displaytext={`${job.progress.current}M/${job.progress.total}M`}
                      />
                    }
                  </StyledTableCell>
                )}
                <StyledTableCell>
                  <a href="#">View</a>
                </StyledTableCell>
              </StyledTableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
};

export default JobTab;
