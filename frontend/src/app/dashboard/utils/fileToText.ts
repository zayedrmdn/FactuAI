/* utils/fileToText.ts */
import { toast } from "sonner";

// NEW 👇 helper that lazy-loads the pre-bundled pdfjs “webpack” build
async function getPdfjs() {
  // this bundle includes the worker, no extra import needed
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  return (await import("pdfjs-dist/webpack")) as typeof import("pdfjs-dist");
}

/** …rest of file stays same … */
export async function fileToText(file: File): Promise<string> {
  const ext = file.name.split(".").pop()?.toLowerCase();

  if (ext === "txt") {
    return await file.text();
  }

  if (ext === "pdf") {
    const pdfjs = await getPdfjs();            // ⬅️ use helper
    const arrayBuf = await file.arrayBuffer();

    const pdf = await pdfjs.getDocument({ data: arrayBuf }).promise;
    let text = "";
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items
        .map((item) => ('str' in item ? item.str : ''))
        .join(" ") + "\n";
    }
    return text;
  }

  toast.error("File type not supported yet. Use TXT or PDF.");
  return "";
}
