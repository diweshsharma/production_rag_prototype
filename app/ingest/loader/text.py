import logfire

def parse_text(file_path: str):
    """
    parses plain text files
    """

    with logfire.span("text parsing", filename = file_path):
        try:
            with open(file_path , 'r' , encoding = 'utf-8' , errors = 'ignore') as f:
                return f.read()

        except Exception as e:
            logfire.error(f"Text Parse Failed : {e}")
            raise e
            