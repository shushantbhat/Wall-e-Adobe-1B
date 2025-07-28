#!/usr/bin/env python3
"""
Simple PDF Reader Demo
Demonstrates reading PDFs from the input directory
"""

import os
import fitz  # PyMuPDF
from pathlib import Path
import json

def read_pdf_content(pdf_path: str) -> dict:
    """Read PDF and extract text content"""
    try:
        doc = fitz.open(pdf_path)
        content = {
            "filename": os.path.basename(pdf_path),
            "pages": len(doc),
            "sections": []
        }
        
        current_section = {"title": "", "content": "", "page": 1}
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        max_font_size = 0
                        is_bold = False
                        
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                line_text += text + " "
                                max_font_size = max(max_font_size, span["size"])
                                if span["flags"] & 2**4:  # Bold flag
                                    is_bold = True
                        
                        line_text = line_text.strip()
                        if not line_text:
                            continue
                        
                        # Detect if this is a heading
                        is_heading = (
                            max_font_size > 12 and is_bold
                        ) or (
                            len(line_text) < 100 and 
                            (line_text.isupper() or line_text.istitle()) and
                            max_font_size > 10
                        )
                        
                        if is_heading and len(line_text) > 3:
                            # Save previous section if it has content
                            if current_section["content"].strip():
                                content["sections"].append(current_section.copy())
                            
                            # Start new section
                            current_section = {
                                "title": line_text,
                                "content": "",
                                "page": page_num + 1
                            }
                        else:
                            # Add to current section content
                            current_section["content"] += line_text + " "
        
        # Add the last section
        if current_section["content"].strip():
            content["sections"].append(current_section)
        
        doc.close()
        return content
        
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return {"filename": os.path.basename(pdf_path), "error": str(e)}

def main():
    """Main function to demonstrate PDF reading"""
    
    input_dir = Path("app/input")
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in input directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    print("\n" + "="*50)
    print("READING PDF CONTENT")
    print("="*50)
    
    all_content = []
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        content = read_pdf_content(str(pdf_file))
        all_content.append(content)
        
        print(f"  Pages: {content.get('pages', 'N/A')}")
        print(f"  Sections found: {len(content.get('sections', []))}")
        
        # Show first few sections
        sections = content.get('sections', [])
        for i, section in enumerate(sections[:3]):  # Show first 3 sections
            print(f"    Section {i+1}: {section.get('title', 'No title')}")
            content_preview = section.get('content', '')[:100]
            if content_preview:
                print(f"      Preview: {content_preview}...")
    
    # Save results
    output_file = Path("app/output/pdf_reading_demo.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_content, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total PDFs processed: {len(all_content)}")

if __name__ == "__main__":
    main() 