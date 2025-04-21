import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

# Example usage:
pdf_path = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition.pdf"  # Change this to your actual PDF file path
extracted_text = extract_text_from_pdf(pdf_path)

# Save extracted text to a file
with open("Galethird.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)
