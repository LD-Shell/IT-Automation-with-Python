# Run with python3 remove_bookmark.py
# Install PyMuPDF first with pip install PyMuPDF
import fitz  # PyMuPDF

input_file = "new.pdf"
output_file = "OChEGS_Sponsorship_Package_2025-2026.pdf"

doc = fitz.open(input_file)

# Remove TOC (table of contents = outlines/bookmarks)
doc.set_toc([])

# Save to new file
doc.save(output_file)
doc.close()
print("✅ Outline removed. Saved as:", output_file)
