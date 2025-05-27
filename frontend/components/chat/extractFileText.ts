// import { extractText } from 'pptx-text';
import pdfToText from 'react-pdftotext';
import mammoth from 'mammoth';
import * as XLSX from 'xlsx';

const extractFileText = async (file: File): Promise<string> => {
    try {
        const extension = file.name.split('.').pop()?.toLowerCase();

        switch (extension) {
            case 'pdf':
                return await pdfToText(file);

            case 'docx':
                const arrayBuffer = await file.arrayBuffer();
                const result = await mammoth.extractRawText({ arrayBuffer });
                return result.value;

            case 'xlsx':
                const buffer = await file.arrayBuffer();
                const workbook = XLSX.read(new Uint8Array(buffer), { type: 'array' });
                let text = '';
                workbook.SheetNames.forEach(sheetName => {
                    const worksheet = workbook.Sheets[sheetName];
                    const sheetText = XLSX.utils.sheet_to_txt(worksheet);
                    text += `[Sheet: ${sheetName}]\n${sheetText}\n\n`;
                });
                return text.trim();

            // case 'pptx':
            //     return await extractText(file);

            case 'txt':
            case 'csv':
                return await file.text();

            default:
                throw new Error('Unsupported file type');
        }
    } catch (error) {
        console.error(`Error extracting text from file: ${file.name}`, error);
        throw error;
    }
};

export default extractFileText;