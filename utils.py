import json
import re

def clean_text(text):
    return text.strip() if text else None


def is_valid_invoice_number(text):
    """
    Filter out wrong numbers like:
    - phone numbers
    - GST numbers
    - amounts
    - random long numbers
    """

    if not text:
        return False

    text = text.strip()

    #  Reject very long numbers (like IRN, GST)
    if len(text) > 20:
        return False

    # Reject extremely short values like 's' or 'No'
    if len(text) < 1:
        return False

    #  Reject pure numbers > 8 digits (likely not invoice no)
    if text.isdigit() and len(text) > 8:
        return False

    # Accept patterns like:
    # NSS/1254, JME/25-26/13042, 8768, 25, LC07
    pattern = r'^[A-Za-z0-9\/\-]+$'
    if not re.match(pattern, text):
        return False

    return True


def normalize_key(key):
    if not key or not isinstance(key, str):
        return None
    return re.sub(r'[^a-z0-9]', '', key.lower())


def extract_value(data, candidates):
    if not isinstance(data, dict):
        return None
    try:
        normalized = {normalize_key(k): v for k, v in data.items() if isinstance(k, str)}
        for candidate in candidates:
            value = normalized.get(normalize_key(candidate))
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception as e:
        print(f"[WARNING] Error extracting value from data: {str(e)}")
    return None


def extract_date_from_text(text):
    if not text or not isinstance(text, str):
        return None
    
    # Look for date patterns
    date_patterns = [
        r'(?:date|invoice date|date of invoice|issued on|inv.*date)\s*[:\-]?\s*(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{1,2}\s(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s\d{4})',
        r'(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})'
    ]
    
    for pattern in date_patterns:
        date_match = re.search(pattern, text, re.IGNORECASE)
        if date_match:
            return clean_text(date_match.group(1))
    
    return None


def extract_invoice_number_from_text(text):
    if not text or not isinstance(text, str):
        return None

    # Clean up common OCR artifacts
    text = text.replace('<', '').replace('>', '').replace('_seinice', '').replace('_invoice', '')
    
    # Prefer explicit invoice labels first
    label_patterns = [
        r'(?:invoice|inv|invoice\s*no|inv\s*no|no|number|No)\s*[:\-]?\s*([A-Za-z0-9]+(?:[\/\- ][A-Za-z0-9]+)*)',
        r'([A-Za-z0-9]+(?:[\/\- ][A-Za-z0-9]+)*)\s*(?:invoice|inv|invoice\s*no|inv\s*no|no|number)'
    ]
    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = clean_text(match.group(1))
            if is_valid_invoice_number(candidate):
                return candidate

    # Fallback: extract tokens that look like invoice numbers
    matches = re.findall(r'[A-Za-z0-9]+(?:[\/\- ][A-Za-z0-9]+)*', text)
    for match in matches:
        cleaned = clean_text(match)
        if cleaned and re.search(r'\d', cleaned) and is_valid_invoice_number(cleaned):
            return cleaned

    return None


def parse_output(result):
    if not result or not isinstance(result, str):
        return None, None

    invoice_no = None
    date = None
    
    # Clean and normalize result
    result = result.strip()
    result = re.sub(r'<\/?s_invoice>|<s>|<\/s>', '', result, flags=re.IGNORECASE).strip()
    
    # Extract JSON if embedded in text
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        # Try to extract JSON object from noisy output
        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        else:
            data = None

    if isinstance(data, dict) and data:
        invoice_no = clean_text(extract_value(data, [
            'invoice_number',
            'invoice no',
            'invoice_no',
            'invoice',
            'Invoice No'
        ]))
        date = clean_text(extract_value(data, [
            'invoice_date',
            'invoice date',
            'date',
            'Invoice Date'
        ]))
    
    # Fallback: Try to extract from raw text
    if not invoice_no:
        invoice_no = extract_invoice_number_from_text(result)

    if not date:
        date = extract_date_from_text(result)

    # Validate invoice number
    if invoice_no and not is_valid_invoice_number(invoice_no):
        invoice_no = None

    return invoice_no, date