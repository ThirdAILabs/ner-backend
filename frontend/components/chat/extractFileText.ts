import pdfToText from 'react-pdftotext';

export default async function extractFileText(file?: File) {
    if (file && file.type === 'application/pdf') {
        return await pdfToText(file);
    }
    throw new Error(`Unsupported file type: ${file?.type}`);
}