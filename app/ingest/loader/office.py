import logfire
from unstructured.partition.auto import partition 

def parse_office(file_path: str):
    """
    parses office documents(.docs , .pptx) using unstructured library.
    Unlike PDFs , these formats are structured , so we can extract text directly.
    """
    with logfire.span("Office Documnet Parsing" , filename = file_path):

        try:
            elements = partition(filename=file_path)
            full_text = "\n".join([str(el) for el in elements])

            if not full_text.strip():
                raise ValueError("No text extracted from office document")
        
            return full_text
        except Exception as e:
            logfire.error(f"Office Parse Failed : {e}")
            raise e
