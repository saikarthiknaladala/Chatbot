import re
from PyPDF2 import PdfReader

def remove_punc(text):
    # Remove headers and footers based on patterns or positions
    # Example: Remove text from the top 10% and bottom 10% of each page
    pattern = r"[^a-zA-Z0-9.,!?/@#*]+"
    # Apply the regex pattern to remove unwanted characters
    cleaned_text = re.sub(pattern, " ", text)
    return cleaned_text

def preprocess_pdf(input_path, output_path):
    start_index = None
    end_index = None
    text_pages = []
    # Open the PDF file
    with open(input_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages) 
        
        processed_text = ''
        for page_num in range(num_pages):
            # Extract text from the page
            page = reader.pages[page_num]
            text = page.extract_text()
            print(text)
            # Remove headers and footers
            text = remove_punc(text)
            text_pages.append(text)

            if 'TABLE OF CONTENTS' in text and start_index is None:
                start_index = page_num
            if 'Index' in text and start_index is not None:
                end_index = page_num
                break

        if start_index is not None and end_index is not None:
            # Remove text between start and end keywords
            for i in range(start_index, end_index + 1):
                text_pages[i] = ''


    processed_text = ''.join(text_pages)            
    
    # Write processed text to a new PDF or text file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(processed_text)

# Example usage
input_file = 'localGPT/SOURCE_DOCUMENTS/2023-2024_UMKC_Catalog.pdf'
output_file = 'localGPT/SOURCE_DOCUMENTS/cleaned_UMKC_catalog.pdf'
preprocess_pdf(input_file, output_file)