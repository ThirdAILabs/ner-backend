import pdfToText from 'react-pdftotext';
import { pdfjs } from 'react-pdf';
import mammoth from 'mammoth';
import * as XLSX from 'xlsx';
// import * as officeParser from 'officegen';

// Set worker source to use local file. See gotchas.md, "pdfjs-dist fixed to 4.8.69"
pdfjs.GlobalWorkerOptions.workerSrc = require('pdfjs-dist/build/pdf.worker.mjs');

export default async function extractFileText(file?: File): Promise<string> {
  if (!file) {
    throw new Error('No file provided');
  }

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
        workbook.SheetNames.forEach((sheetName) => {
          const worksheet = workbook.Sheets[sheetName];
          const sheetText = XLSX.utils.sheet_to_txt(worksheet);
          text += `[Sheet: ${sheetName}]\n${sheetText}\n\n`;
        });
        return text.trim();

      // case 'pptx':
      //     const pptxBuffer = await file.arrayBuffer();
      //     const pptxResult = await new Promise<string>((resolve, reject) => {
      //         officeParser.load(pptxBuffer, (err: any, data: any) => {
      //             if (err) reject(err);
      //             let text = '';
      //             data.slides.forEach((slide: any, index: number) => {
      //                 text += `[Slide ${index + 1}]\n`;
      //                 slide.objects.forEach((obj: any) => {
      //                     if (obj.type === 'text') {
      //                         text += obj.text + '\n';
      //                     }
      //                 });
      //                 text += '\n';
      //             });
      //             resolve(text.trim());
      //         });
      //     });
      //     return pptxResult;

      case 'txt':
      case 'csv':
        return await file.text();

      default:
        throw new Error(`Unsupported file type: ${file.type}`);
    }
  } catch (error) {
    console.error(`Error extracting text from file: ${file.name}`, error);
    throw error;
  }
}
