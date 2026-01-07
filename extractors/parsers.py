import re

def parse_japanese_id(ocr_results):
    """
    Parses OCR output for Japanese Resident Cards (Zairyu Card).
    """
    data = {
        "doc_type": "Japanese Resident Card",
        "name": None,
        "birth_date": None,
        "card_id": None,
        "address": None,
        "raw_text": []
    }
    
    if not ocr_results or ocr_results[0] is None:
        return data

    lines = [line[1][0] for line in ocr_results[0]]
    data["raw_text"] = lines

    # Regex Patterns
    # 12 digit ID usually at top right (Alphanumeric)
    id_pattern = re.compile(r'[A-Za-z]{2}\d{8}[A-Za-z]{2}') 
    # Date pattern (YYYY年MM月DD日)
    date_pattern = re.compile(r'\d{4}年\d{1,2}月\d{1,2}日')
    
    for i, line in enumerate(lines):
        # Finding ID
        if id_pattern.search(line):
            data["card_id"] = id_pattern.search(line).group()

        # Finding Name (Simple heuristic: Line after '氏名')
        if "氏名" in line:
            # Check if name is on same line
            clean_line = line.replace("氏名", "").strip()
            if len(clean_line) > 1:
                data["name"] = clean_line
            # Else check next line
            elif i + 1 < len(lines):
                data["name"] = lines[i+1]

        # Finding DOB
        if "生年月日" in line:
             match = date_pattern.search(line)
             if match:
                 data["birth_date"] = match.group()
             elif i + 1 < len(lines):
                 match = date_pattern.search(lines[i+1])
                 if match: data["birth_date"] = match.group()

        # Finding Address
        if "住居地" in line:
            clean_line = line.replace("住居地", "").strip()
            if len(clean_line) > 1:
                data["address"] = clean_line
            elif i + 1 < len(lines):
                data["address"] = lines[i+1]

    return data

def parse_brunei_id(ocr_results):
    """
    Parses OCR output for Brunei Identity Cards.
    """
    data = {
        "doc_type": "Brunei Identity Card",
        "ic_number": None,
        "name": None,
        "dob": None,
        "raw_text": []
    }
    
    if not ocr_results or ocr_results[0] is None:
        return data

    lines = [line[1][0] for line in ocr_results[0]]
    data["raw_text"] = lines

    # Regex
    # Brunei IC format often: 00-000000 (2 digits - 6 digits)
    ic_pattern = re.compile(r'\d{2}-\d{6}')
    
    for i, line in enumerate(lines):
        if ic_pattern.search(line):
            data["ic_number"] = ic_pattern.search(line).group()
            
        if "Nama" in line or "Name" in line:
            if i + 1 < len(lines):
                data["name"] = lines[i+1]

    return data

def master_parser(ocr_results):
    """
    Determines document type and routes to correct parser.
    """
    if not ocr_results or ocr_results[0] is None:
        return {}

    # Flatten text to check keywords
    full_text = " ".join([line[1][0] for line in ocr_results[0]])
    
    if "日本" in full_text or "在留" in full_text:
        return parse_japanese_id(ocr_results)
    elif "Brunei" in full_text or "Negara" in full_text or "K/P" in full_text:
        return parse_brunei_id(ocr_results)
    else:
        # Default fallback
        return {"doc_type": "Unknown", "raw_text": [line[1][0] for line in ocr_results[0]]}