import pdfplumber
import unicodedata
import tiktoken
import re
import difflib


# maintains better line breaks, column layouts, and overall fidelity to original format.
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''.join(page.extract_text() or '' for page in pdf.pages)
    return text

def clean_text(text):
    return unicodedata.normalize('NFKC', text)

def count_tokens(text):
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def chunk_text(text, max_tokens=512):
    encoder = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        word_tokens = len(encoder.encode(word + " "))
        if current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def fuzzy_match_header(header, valid_headers):
    """
    Tries to match a given header string against a list of valid headers.
    Returns the best match if confidence is high enough.
    """
    match = difflib.get_close_matches(header, valid_headers, n=1, cutoff=0.7)
    return match[0] if match else None

def extract_sections(text, section_headers=None, max_tokens=512):
    
    if section_headers is None:
        section_headers = ['Abstract', 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion']

    # Normalize newlines
    text = re.sub(r'\n+', '\n', text)

    # Find all potential section headers (lines with capital words)
    lines = text.split('\n')
    potential_headers = [(i, line.strip()) for i, line in enumerate(lines) if len(line.strip()) < 100]

    header_indices = []
    for i, line in potential_headers:
        match = fuzzy_match_header(line.strip().title(), section_headers)
        if match:
            header_indices.append((i, match))

    # If no headers found, fallback to abstract guess
    if not header_indices:
        fallback_text = " ".join(text.split()[:250])
        chunks = chunk_text(fallback_text, max_tokens)
        return {
            "Abstract": [{"chunk": chunk, "token_count": count_tokens(chunk)} for chunk in chunks]
        }

    sections = {}
    for i, (line_index, header_name) in enumerate(header_indices):
        start = line_index
        end = header_indices[i + 1][0] if (i + 1) < len(header_indices) else len(lines)
        section_text = "\n".join(lines[start+1:end]).strip()
        chunks = chunk_text(section_text, max_tokens)
        sections[header_name] = [{"chunk": chunk, "token_count": count_tokens(chunk)} for chunk in chunks]

    return sections

