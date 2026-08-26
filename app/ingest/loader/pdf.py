import logfire

from pypdf import PdfReader

def parse_pdf(file_path: str) -> str :
    """
    parses pdf documents using pypdf library.
    """
    with logfire.span("PDF Parsing" , filename = file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"Total Pages : {total_pages}")

            text_parts : list[str] = []
            blank_pages : list[str] = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
                else:
                    blank_pages.append(i + 1)

            if blank_pages:
                logfire.info(f"Blank pages detected {blank_pages} retrying with pdfplumber")
                try:
                    import pdfplumber 
                    with pdfplumber.open(file_path) as pdf:
                        for page_num in blank_pages:
                            page = pdf.pages[page_num - 1]
                            fallback_text = page.extract_text() or ""
                            if fallback_text.strip():
                                text_parts.append(fallback_text)

                except Exception as e:
                    logfire.warning(f"Pdfplumber fallback failed : {e}")

            if not text_parts:
                raise ValueError("No text extracted from PDF")

            return "\n\n".join(text_parts)

        except Exception as e:
            logfire.error(f"PDF Parse Failed : {e}")
            raise e