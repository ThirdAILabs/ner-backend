import pdfToText from 'react-pdftotext';
import { pdfjs } from 'react-pdf';

// Set worker source to use local file. See gotchas.md, "pdfjs-dist fixed to 4.8.69"
pdfjs.GlobalWorkerOptions.workerSrc = require('pdfjs-dist/build/pdf.worker.mjs');

export default async function extractFileText(file?: File) {
    if (file && file.type === 'application/pdf') {
        return await pdfToText(file);
    }
    throw new Error(`Unsupported file type: ${file?.type}`);
}