import React from 'react';
import { styled } from '@mui/material/styles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell, { tableCellClasses } from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import '../../../styles/pages/modelpage/job_tab.scss';

interface DocumentItem {
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

const JobTab: React.FC = () => {
    const documents: DocumentItem[] = [
        {
          name: 'Daniel Docs 1',
          source: 'S3 Bucket - Daniel Docs 1',
          initiated: '2 months ago',
          progress: {
            status: 'Completed',
            completed: '2 months ago'
          }
        },
        {
          name: 'Daniel Docs 2',
          source: 'S3 Bucket - Daniel Docs 2',
          initiated: '3 days ago',
          progress: {
            status: 'InProgress',
            current: 52,
            total: 92
          }
        }
      ];

    const renderProgress = (progress: DocumentItem['progress']) => {
        if (progress.status === 'Completed') {
          return <span>Completed {progress.completed}</span>;
        } else {
          return (
            <div className="progress-container">
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${(progress.current! / progress.total!) * 100}%` }}
                ></div>
              </div>
              <span className="progress-text">{progress.current}M / {progress.total}M</span>
            </div>
          );
        }
      };

    return (
      <div className='job-tab'>
        <button>
          New
        </button>
        <TableContainer component={Paper}>
                <Table sx={{ minWidth: 700 }} aria-label="customized table">
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
                        {documents.map((doc, index) => (
                            <StyledTableRow key={index}>
                                <StyledTableCell>{doc.name}</StyledTableCell>
                                <StyledTableCell>{doc.source}</StyledTableCell>
                                <StyledTableCell>{doc.initiated}</StyledTableCell>
                                <StyledTableCell>{renderProgress(doc.progress)}</StyledTableCell>
                                <StyledTableCell>
                                    <a href="#" className="view-link">View</a>
                                </StyledTableCell>
                            </StyledTableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
      </div>
            
    );
}

export default JobTab;