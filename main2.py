import os
import shutil
import logging
import uuid
import json
import re
import zipfile
import subprocess
from typing import List, Dict, Any
from pathlib import Path
import asyncio
from datetime import datetime
import latex2mathml.converter
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import google.generativeai as genai
import markdown
from markdown.extensions import codehilite, tables,toc
import markdown2
import markdown
from markdown.extensions import codehilite, tables, toc
import markdown
# For better PDF generation with LaTeX support
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import tempfile
import base64   
from fastapi.responses import StreamingResponse
import io
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")  # Set your API key here
    UPLOAD_DIR = Path("uploads")
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.pptx', '.md', '.py', '.cpp', '.java', '.html', '.css', '.js'}

# Initialize Gemini
if Config.GOOGLE_API_KEY:
    genai.configure(api_key=Config.GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")  # Using latest model
else:
    logger.warning("GOOGLE_API_KEY not set!")

app = FastAPI(title="Beautiful Note Maker", version="2.0")
Config.UPLOAD_DIR.mkdir(exist_ok=True)

class NoteGenerator:
    async def generate_notes_from_file(self, filepath: str) -> str:
        """Generate beautiful notes from any file type."""
        try:
            logger.info(f"Processing file: {filepath}")
            
            # Upload file to Gemini
            uploaded_file = genai.upload_file(filepath)
            logger.info("File uploaded to Gemini")
            
            # Generate notes with enhanced prompt
            response = model.generate_content(
                [uploaded_file,""" You are an expert note-maker specializing in converting video lecture transcripts and ppt into beautiful, structured, short and concise study notes .
Give notes in short points instead of large paragraph.
**CRITICAL MATH FORMATTING RULE**:
- ALWAYS keep math formulas on ONE single line
- NEVER break formulas across multiple lines
- WRONG: $\frac{d}{dw} J(w)
$
- CORRECT: $\frac{d}{dw} J(w)$

**MATHEMATICAL FORMULA ENHANCEMENT**:
When you encounter mathematical expressions, formulas, or equations:

1. **Keep ALL math on single lines**
2. **Use proper LaTeX notation** with appropriate spacing
3. **Add descriptive text** before complex formulas


**ADVANCED MATH FORMATTING RULES**:
- **Derivatives**: Use `$\frac{d}{dx}f(x)$` (all on one line)
- **Functions**: Use `$f(x) = wx + b$` (all on one line)  
- **Complex formulas**: Use `$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(f(x^{(i)}) - y^{(i)})^2$$` (all on one line)

**FORMULA PRESENTATION**:
- Put important formulas on separate lines but keep the LaTeX on one line
- Add context before showing the formula

**EXAMPLE FORMAT**:
                 J(w,b)=(1/2m)∗sum((f(x)−y)2)
**FORMATTING RULES**:
- Use `#` for main topics, `##` for sections, `###` for subsections
- **Bold** important terms and definitions  
- *Italicize* for emphasis
- Use `>` for key insights and important quotes
- Create bullet points for lists and examples
- Add `---` to separate major sections

**MATHEMATICAL NOTATION RULES** (VERY IMPORTANT):
- For inline math (short formulas): `$E=mc^2$` or `$x^2 + y^2 = z^2$`
- For display math (long formulas): Put on separate lines with blank lines before/after:

**CRITICAL MATH FORMATTING**:
- NEVER break math formulas across multiple lines
- ALWAYS put complex formulas on their own line
- Use blank lines before and after display math
- Keep each formula complete in one $$...$$ block                 
                 
**CONTENT STRUCTURE**:
1. Start with the main topic as `# [Topic Name]`
2. Organize content into logical sections
3. Include examples and explanations from the transcript
4. Extract key formulas, definitions, and concepts
5. Remove filler words, "um", "uh", repetitions

**OUTPUT**: Clean, well-structured Markdown notes that a student would want to study from."""],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=8192,
                )
            )
            
            logger.info("Notes generated successfully")
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating notes for {filepath}: {str(e)}")
            return f"⚠️ **Error Processing File**: {filepath}\n\n```\n{str(e)}\n```"

class PDFGenerator:
    def __init__(self):
        self.font_config = FontConfiguration()
        
    def create_beautiful_pdf(self, notes_dict: Dict[str, str], output_file: str = "beautiful_notes.pdf") -> str:
        """Create a beautiful PDF with proper LaTeX math rendering."""
        try:
            # Merge all notes
            merged_notes = self._merge_notes(notes_dict)
            
            # Convert markdown to HTML with math support
            html_content = self._markdown_to_html(merged_notes)
            
            # Add beautiful CSS styling
            css_styles = self._get_beautiful_css()
            
            # Generate PDF using WeasyPrint (better LaTeX support)
            output_path = Config.UPLOAD_DIR / output_file
            HTML(string=html_content).write_pdf(
                output_path,
                stylesheets=[CSS(string=css_styles)],
                font_config=self.font_config
            )
            
            logger.info(f"PDF generated successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            return self._fallback_pdf_generation(notes_dict, output_file)
    
    def _merge_notes(self, notes_dict: Dict[str, str]) -> str:
        """Merge notes from multiple files with beautiful formatting."""
        timestamp = datetime.now().strftime("%B %d, %Y")
        
        merged_content = f"""# 🎓 Comprehensive Study Notes
*Generated on {timestamp}*

---

## 📋 Table of Contents
"""
        
        # Add table of contents
        for i, (filename, _) in enumerate(notes_dict.items(), 1):
            clean_name = filename.replace('.', '_').replace(' ', '_')
            merged_content += f"{i}. [{filename}](#{clean_name})\n"
        
        merged_content += "\n---\n\n"
        
        # Add content from each file
        for filename, notes in notes_dict.items():
            clean_name = filename.replace('.', '_').replace(' ', '_')
            merged_content += f"""
## 📄 {filename} {{#{clean_name}}}

{notes}

---

"""
        
        return merged_content
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        import markdown
        from katex import render
    
        # --- Replace inline math $...$ ---
        def replace_inline_math(match):
            latex = match.group(1).strip()
            try:
                return render(latex, display_mode=False)
            except Exception as e:
                return f"<code>{latex}</code>"
    
        # --- Replace display math $$...$$ ---
        def replace_display_math(match):
            latex = match.group(1).strip()
            try:
                return render(latex, display_mode=True)
            except Exception as e:
                return f"<pre>{latex}</pre>"
    
        # Replace math expressions with KaTeX HTML
        markdown_text = re.sub(r"\$\$(.+?)\$\$", replace_display_math, markdown_text, flags=re.DOTALL)
        markdown_text = re.sub(r"\$(.+?)\$", replace_inline_math, markdown_text)
    
        # Convert Markdown → HTML
        md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
        html_content = md.convert(markdown_text)
    
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Beautiful Study Notes</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
            <style>
                body {{
                    font-family: 'Inter', sans-serif;
                    line-height: 1.6;
                    color: #2d3748;
                    max-width: 190mm;
                    margin: 0 auto;
                    padding: 15mm 10mm;
                    background: #ffffff;
                }}
                .katex-display {{
                    margin: 1em 0;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """

    
    def _get_beautiful_css(self) -> str:
        """Return beautiful CSS styles for the PDF."""
        return """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    line-height: 1.6;
    color: #2d3748;
    max-width: 190mm;  /* Reduced from 210mm */
    margin: 0 auto;
    padding: 15mm 10mm;  /* Reduced left/right padding from 20mm */
    background: #ffffff;
}

/* Headings - REDUCED SIZES */
h1 {
    font-size: 1.8em;  /* Reduced from 2.5em */
    color: #1a202c;
    border-bottom: 3px solid #4299e1;
    padding-bottom: 0.3em;  /* Reduced padding */
    margin-top: 1.5em;  /* Reduced margins */
    margin-bottom: 0.8em;
    font-weight: 700;
}

h2 {
    font-size: 1.4em;  /* Reduced from 2em */
    color: #2d3748;
    margin-top: 1.2em;  /* Reduced margins */
    margin-bottom: 0.6em;
    font-weight: 600;
    border-left: 4px solid #4299e1;
    padding-left: 0.8em;  /* Reduced padding */
}

h3 {
    font-size: 1.2em;  /* Reduced from 1.5em */
    color: #4a5568;
    margin-top: 1em;
    margin-bottom: 0.5em;
    font-weight: 500;
}

h4 {
    font-size: 1.1em;  /* Reduced from 1.2em */
    color: #718096;
    margin-top: 0.8em;
    margin-bottom: 0.4em;
    font-weight: 500;
}

/* Text formatting */
p {
    margin-bottom: 1em;
    text-align: justify;
    margin-left: 0;  /* Remove any left margin */
    margin-right: 0; /* Remove any right margin */
}

strong {
    color: #1a202c;
    font-weight: 600;
}

em {
    color: #4a5568;
}
/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace;
    background: #f7fafc;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    color: #e53e3e;
    font-size: 0.9em;
}

pre {
    background: #1a202c;
    color: #f7fafc;
    padding: 1.2em;  /* Reduced padding */
    border-radius: 8px;
    overflow-x: auto;
    margin: 1.2em 0;  /* Reduced margins */
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.2em 0;  /* Reduced margins */
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

th, td {
    border: 1px solid #e2e8f0;
    padding: 0.6em;  /* Reduced padding */
    text-align: left;
}

th {
    background: #4299e1;
    color: white;
    font-weight: 600;
}

tr:nth-child(even) {
    background: #f8f9fa;
}

/* Blockquotes */
blockquote {
    border-left: 4px solid #38b2ac;
    background: #e6fffa;
    padding: 0.8em 1.2em;  /* Reduced padding */
    margin: 1.2em 0;  /* Reduced margins */
    border-radius: 0 8px 8px 0;
}

blockquote p {
    margin: 0;
    color: #234e52;
}

/* Lists */
ul, ol {
    margin: 1em 0;
    padding-left: 1.5em;  /* Reduced from 2em */
}

li {
    margin: 0.4em 0;  /* Reduced margins */
}

/* Math styling */
.MathJax {
    font-size: 1.1em !important;
    margin: 1em 0 !important;
    display: block !important;
    text-align: center !important;
}

.MathJax_Display {
    margin: 1.2em 0 !important;  /* Reduced margins */
    text-align: center !important;
}

/* Ensure math blocks don't break */
.math {
    page-break-inside: avoid !important;
    margin: 1em 0 !important;
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 2px solid #e2e8f0;
    margin: 1.5em 0;  /* Reduced margins */
}

/* Page breaks */
.page-break {
    page-break-before: always;
}

/* Additional content width optimization */
.content {
    width: 100%;
    max-width: none;
}

/* Math styling - ADD THIS */
.MathJax {
    font-size: 1.1em !important;
    margin: 1em 0 !important;
    display: block !important;
    text-align: center !important;
}

.MathJax_Display {
    margin: 1.5em 0 !important;
    text-align: center !important;
}

/* Ensure math blocks don't break */
.math {
    page-break-inside: avoid !important;
    margin: 1em 0 !important;
}
/* Basic MathML support */
math {
    font-family: 'Latin Modern Math', 'STIX Two Math', serif;
    font-size: 1.1em;
    display: inline-block;
    margin: 0 0.2em;
}

math[display="block"] {
    display: block;
    text-align: center;
    margin: 1em 0;
}

"""
    
    def _fallback_pdf_generation(self, notes_dict: Dict[str, str], output_file: str) -> str:
        """Fallback PDF generation using ReportLab if WeasyPrint fails."""
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        
        output_path = Config.UPLOAD_DIR / output_file
        doc = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=1*inch)
        
        # Enhanced styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=HexColor('#1a202c'),
            spaceAfter=30
        ))
        
        story = []
        story.append(Paragraph("🎓 Comprehensive Study Notes", styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        for filename, notes in notes_dict.items():
            story.append(Paragraph(f"📄 {filename}", styles['Heading1']))
            story.append(Spacer(1, 0.2*inch))
            
            # Simple markdown-to-paragraph conversion
            lines = notes.split('\n')
            for line in lines:
                if line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                    story.append(Spacer(1, 0.1*inch))
            
            story.append(PageBreak())
        
        doc.build(story)
        return str(output_path)

# Initialize components
note_generator = NoteGenerator()
pdf_generator = PDFGenerator()


@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    """Upload and process a ZIP file to generate beautiful notes."""
    try:
        # Validate file
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
        
        if file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Create unique directory for this upload
        unique_id = str(uuid.uuid4())
        upload_dir = Config.UPLOAD_DIR / unique_id
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_location = upload_dir / file.filename
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract ZIP
        extract_path = upload_dir / "extracted"
        with zipfile.ZipFile(file_location, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        
        logger.info(f"Files extracted to {extract_path}")
        
        # Process all files in numerical order
        notes_dict = {}
        processed_count = 0

        # Collect all subtitle files first
        subtitle_files = []
        other_files = []

        for root, _, files in os.walk(extract_path):
            for fname in files:
                file_path = Path(root) / fname
                
                # Check if file extension is allowed
                if file_path.suffix.lower() in Config.ALLOWED_EXTENSIONS:
                    # Check if it's a subtitle file with pattern subtitle(n)
                    if re.match(r'subtitle\(\d+\)', fname.lower()):
                        subtitle_files.append((file_path, fname))
                    else:
                        other_files.append((file_path, fname))

        # Sort subtitle files numerically by the number in parentheses
        def extract_subtitle_number(filename):
            match = re.search(r'subtitle\((\d+)\)', filename.lower())
            return int(match.group(1)) if match else 0

        subtitle_files.sort(key=lambda x: extract_subtitle_number(x[1]))

        # Process subtitle files first in order, then other files
        all_files = subtitle_files + other_files

        for file_path, fname in all_files:
            try:
                logger.info(f"Processing {fname}...")
                notes = await note_generator.generate_notes_from_file(str(file_path))
                notes_dict[fname] = notes
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing {fname}: {str(e)}")
                notes_dict[fname] = f"⚠️ **Error processing {fname}**: {str(e)}"        
        if not notes_dict:
            raise HTTPException(status_code=400, detail="No processable files found in ZIP")
        
        # Generate beautiful PDF
        pdf_filename = f"beautiful_notes_{unique_id}.pdf"
        pdf_bytes = io.BytesIO()
        pdf_path = pdf_generator.create_beautiful_pdf(notes_dict, "temp.pdf")

        with open(pdf_path, "rb") as f:
            pdf_bytes.write(f.read())

        pdf_bytes.seek(0)

        return StreamingResponse(
            pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=beautiful_notes.pdf"}
        )

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated PDF file."""
    file_path = Config.UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/pdf'
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    }

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    print("🚀 Starting Beautiful Note Maker API...")
    print(f"📁 Upload directory: {Config.UPLOAD_DIR}")
    print("🔗 API will be available at: http://localhost:8000")
    print("📚 Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
