**pdfjs-dist fixed to 4.8.69**  
We encountered this error: `Failed to extract text from PDF: Error: Setting up worker failed: "Failed to fetch dynamically imported module: app://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.8.69/pdf.worker.mjs`
Interpretation: it tries to download a worker script from cloudflare but for some reason the url protocol changes from `http:` to `app:`. We can avoid the download entirely by pinning pdfjs-dist to 4.8.69, the version expected by react-pdftotext. By installing the same version, we can use the worker script in installed in `node_modules/pdfjs-dist/build/pdf.worker.mjs` instead of downloading from cloudflare. See `frontend/components/chat/extractFileText.ts`

