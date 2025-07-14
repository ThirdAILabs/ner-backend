from markitdown import MarkItDown
import tempfile
import os

def parse(file_name, file_contents):
    """Parse file contents using markitdown and return the converted markdown text."""
    try:
        # Initialize MarkItDown
        md = MarkItDown()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=file_name) as temp_file:
            temp_file.write(file_contents)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            # Convert the temporary file
            result = md.convert(temp_path)
            return result.text_content if hasattr(result, 'text_content') else str(result)
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        return f"Error converting file: {str(e)}"