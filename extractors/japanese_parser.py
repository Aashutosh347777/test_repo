"""
Japanese Document Field Extraction Module
EKYC Verification System
Demo / Evaluation Only - Not Production Grade

Supported Documents:
1. Residence Card (在留カード) - Foreign residents in Japan
   - Card Number: XX00000000XX format
   - Fields: Name, DOB, Nationality, Status of Residence, Period of Stay, etc.

2. My Number Card (マイナンバーカード) - Japanese National ID
   - My Number: 12-digit number
   - Fields: Name, DOB, Gender, Address, Expiry Date

3. Driving License (運転免許証) - Japanese Driver's License
   - License Number: 12-digit format
   - Fields: Name, DOB, Address, Issue Date, Expiry Date, License Categories

Extraction Rules:
- Extraction is performed ONLY from raw OCR text
- No normalization of dates or names
- Handles both Japanese and Romanized text
- If a field is not found -> return null
"""

import re
import unicodedata
from typing import Optional, Dict, List


class JapaneseFieldExtractor:
    """
    Rule-based field extractor for Japanese documents.
    Deterministic extraction only - no ML, no inference.
    """

    # Document type constants
    DOC_RESIDENCE_CARD = "RESIDENCE_CARD"
    DOC_MYNUMBER_CARD = "MYNUMBER_CARD"
    DOC_DRIVING_LICENSE = "DRIVING_LICENSE"

    # Residence Card fields (在留カード)
    RESIDENCE_CARD_FIELDS = [
        "full_name",
        "full_name_japanese",
        "date_of_birth",
        "nationality",
        "region",
        "gender",
        "status_of_residence",
        "period_of_stay",
        "work_restriction",
        "card_number",
        "issue_date",
        "expiry_date",
        "address"
    ]

    # My Number Card fields (マイナンバーカード)
    MYNUMBER_CARD_FIELDS = [
        "full_name",
        "full_name_romaji",
        "date_of_birth",
        "gender",
        "address",
        "my_number",
        "issue_date",
        "expiry_date",
        "digital_cert_expiry"
    ]

    # Driving License fields (運転免許証)
    DRIVING_LICENSE_FIELDS = [
        "full_name",
        "date_of_birth",
        "address",
        "license_number",
        "issue_date",
        "expiry_date",
        "license_categories",
        "conditions",
        "issuing_authority"
    ]

    # Gender mapping
    GENDER_MAP = {
        "男": "MALE",
        "女": "FEMALE",
        "男性": "MALE",
        "女性": "FEMALE",
        "M": "MALE",
        "F": "FEMALE",
        "Male": "MALE",
        "Female": "FEMALE"
    }

    # Common nationalities in Japanese
    NATIONALITY_MAP = {
        "中国": "CHINA",
        "韓国": "SOUTH KOREA",
        "フィリピン": "PHILIPPINES",
        "ベトナム": "VIETNAM",
        "ブラジル": "BRAZIL",
        "ネパール": "NEPAL",
        "インドネシア": "INDONESIA",
        "台湾": "TAIWAN",
        "タイ": "THAILAND",
        "米国": "USA",
        "アメリカ": "USA",
        "インド": "INDIA",
        "ミャンマー": "MYANMAR",
        "バングラデシュ": "BANGLADESH",
        "スリランカ": "SRI LANKA",
        "パキスタン": "PAKISTAN",
        "ペルー": "PERU"
    }

    # Status of Residence types (在留資格)
    RESIDENCE_STATUS_LIST = [
        "技術・人文知識・国際業務",
        "技術·人文知識·国際業務",  # OCR variant
        "特定技能1号",
        "特定技能2号",
        "技能実習1号",
        "技能実習2号",
        "技能実習3号",
        "留学",
        "家族滞在",
        "日本人の配偶者等",
        "永住者の配偶者等",
        "永住者",
        "定住者",
        "特別永住者",
        "技能",
        "経営・管理",
        "企業内転勤",
        "研究",
        "教育",
        "高度専門職1号",
        "高度専門職2号",
        "介護",
        "特定活動"
    ]

    def __init__(self):
        """Initialize extractor with compiled patterns."""
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        # Residence Card number: XX00000000XX (2 letters, 8 digits, 2 letters)
        self.residence_card_pattern = re.compile(r'[A-Z]{2}\d{8}[A-Z]{2}')
        
        # My Number: 12 consecutive digits
        self.mynumber_pattern = re.compile(r'\b\d{12}\b')
        
        # License number: Various formats (typically 12 digits with possible hyphens)
        self.license_number_pattern = re.compile(r'\b\d{12}\b|\b\d{2}-\d{2}-\d{6}-\d{2}\b')
        
        # Japanese date patterns
        self.jp_date_pattern = re.compile(
            r'(\d{4}年\d{1,2}月\d{1,2}日)|'  # 2024年01月15日
            r'(令和\d{1,2}年\d{1,2}月\d{1,2}日)|'  # 令和6年1月15日
            r'(平成\d{1,2}年\d{1,2}月\d{1,2}日)|'  # 平成31年1月1日
            r'(昭和\d{1,2}年\d{1,2}月\d{1,2}日)'   # 昭和64年1月7日
        )

    def normalize_text(self, text: str) -> str:
        """
        Normalize Japanese text for consistent extraction.
        - Convert full-width to half-width where appropriate
        - Normalize Unicode
        """
        if not text:
            return ""
        # NFKC normalization converts full-width alphanumeric to half-width
        text = unicodedata.normalize("NFKC", text)
        # Clean up multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def detect_document_type(self, raw_text: str) -> str:
        """
        Detect Japanese document type from OCR text.
        
        Args:
            raw_text: Raw OCR text
            
        Returns:
            Document type constant
        """
        if not raw_text:
            return self.DOC_RESIDENCE_CARD  # Default
        
        text = self.normalize_text(raw_text).upper()
        text_lower = raw_text.lower()
        
        # Scoring system for detection
        scores = {
            self.DOC_RESIDENCE_CARD: 0,
            self.DOC_MYNUMBER_CARD: 0,
            self.DOC_DRIVING_LICENSE: 0
        }
        
        # Residence Card indicators
        residence_keywords = [
            "在留カード", "RESIDENCE CARD", "在留資格", "在留期間",
            "就労制限", "WORK RESTRICTION", "国籍・地域", "NATIONALITY"
        ]
        for kw in residence_keywords:
            if kw in raw_text or kw.lower() in text_lower:
                scores[self.DOC_RESIDENCE_CARD] += 2
        
        # Check for residence card number format
        if self.residence_card_pattern.search(raw_text):
            scores[self.DOC_RESIDENCE_CARD] += 5
        
        # My Number Card indicators
        mynumber_keywords = [
            "個人番号", "マイナンバー", "INDIVIDUAL NUMBER", "個人番号カード",
            "署名用電子証明書", "利用者証明用電子証明書"
        ]
        for kw in mynumber_keywords:
            if kw in raw_text or kw.lower() in text_lower:
                scores[self.DOC_MYNUMBER_CARD] += 2
        
        # Driving License indicators
        license_keywords = [
            "運転免許証", "免許証番号", "免許の条件", "公安委員会",
            "普通", "中型", "大型", "二輪", "原付"
        ]
        for kw in license_keywords:
            if kw in raw_text:
                scores[self.DOC_DRIVING_LICENSE] += 2
        
        # Return highest scoring type
        return max(scores, key=scores.get)

    def extract(self, raw_text: str, document_type: Optional[str] = None) -> dict:
        """
        Extract fields from Japanese document.
        
        Args:
            raw_text: Raw OCR text (front + back combined)
            document_type: Optional document type override
            
        Returns:
            dict with extracted fields
        """
        if not raw_text:
            return self._empty_result(document_type or self.DOC_RESIDENCE_CARD)
        
        # Auto-detect if not specified
        if not document_type:
            document_type = self.detect_document_type(raw_text)
        
        # Route to appropriate extractor
        if document_type == self.DOC_RESIDENCE_CARD:
            return self.extract_residence_card(raw_text)
        elif document_type == self.DOC_MYNUMBER_CARD:
            return self.extract_mynumber_card(raw_text)
        elif document_type == self.DOC_DRIVING_LICENSE:
            return self.extract_driving_license(raw_text)
        else:
            return self._empty_result(document_type)

    def extract_residence_card(self, raw_text: str) -> dict:
        """
        Extract fields from Japanese Residence Card (在留カード).
        
        Front side contains:
        - Name (Romaji and Japanese)
        - Date of birth
        - Nationality/Region
        - Gender
        - Status of residence
        - Period of stay
        - Photo
        
        Back side contains:
        - Address
        - Work restriction status
        - Card number
        - Expiry date
        """
        text = self.normalize_text(raw_text)
        
        return {
            "document_type": self.DOC_RESIDENCE_CARD,
            "full_name": self._extract_romaji_name(text),
            "full_name_japanese": self._extract_japanese_name(text),
            "date_of_birth": self._extract_date_field(text, ["生年月日", "DATE OF BIRTH"]),
            "nationality": self._extract_nationality(text),
            "region": self._extract_region(text),
            "gender": self._extract_gender(text),
            "status_of_residence": self._extract_residence_status(text),
            "period_of_stay": self._extract_period_of_stay(text),
            "work_restriction": self._extract_work_restriction(text),
            "card_number": self._extract_residence_card_number(text),
            "issue_date": self._extract_date_field(text, ["交付年月日", "DATE OF ISSUE"]),
            "expiry_date": self._extract_date_field(text, ["有効期限", "DATE OF EXPIRY", "まで有効"]),
            "address": self._extract_address(text)
        }

    def extract_mynumber_card(self, raw_text: str) -> dict:
        """
        Extract fields from Japanese My Number Card (マイナンバーカード).
        
        Front side contains:
        - Name (Japanese and Romaji for foreigners)
        - Date of birth
        - Gender
        - Address
        - Expiry date
        - Photo
        
        Back side contains:
        - My Number (12 digits)
        - QR code
        """
        text = self.normalize_text(raw_text)
        
        return {
            "document_type": self.DOC_MYNUMBER_CARD,
            "full_name": self._extract_japanese_name_mynumber(text),
            "full_name_romaji": self._extract_romaji_name(text),
            "date_of_birth": self._extract_date_field(text, ["生年月日"]),
            "gender": self._extract_gender(text),
            "address": self._extract_address_mynumber(text),
            "my_number": self._extract_mynumber(text),
            "issue_date": self._extract_date_field(text, ["発行"]),
            "expiry_date": self._extract_date_field(text, ["有効期限", "まで有効"]),
            "digital_cert_expiry": self._extract_date_field(text, ["署名用電子証明書", "電子証明書の有効期限"])
        }

    def extract_driving_license(self, raw_text: str) -> dict:
        """
        Extract fields from Japanese Driving License (運転免許証).
        
        Fields include:
        - Name
        - Date of birth
        - Address
        - License number
        - Issue date / Expiry date
        - License categories (普通, 中型, 大型, etc.)
        - Conditions (眼鏡等)
        - Issuing authority (公安委員会)
        """
        text = self.normalize_text(raw_text)
        
        return {
            "document_type": self.DOC_DRIVING_LICENSE,
            "full_name": self._extract_japanese_name_license(text),
            "date_of_birth": self._extract_date_field(text, ["生年月日"]),
            "address": self._extract_address_license(text),
            "license_number": self._extract_license_number(text),
            "issue_date": self._extract_date_field(text, ["交付", "交付年月日"]),
            "expiry_date": self._extract_date_field(text, ["有効期限", "まで有効"]),
            "license_categories": self._extract_license_categories(text),
            "conditions": self._extract_license_conditions(text),
            "issuing_authority": self._extract_issuing_authority(text)
        }

    # ========== Residence Card Extraction Methods ==========

    def _extract_romaji_name(self, text: str) -> Optional[str]:
        """Extract Romanized name (typically all caps)."""
        patterns = [
            # Name after NAME label on its own line
            r"(?:NAME)\s*\n\s*([A-Z][A-Z\s\-\.]+?)(?=\s*\n|$)",
            # Name after explicit labels
            r"(?:NAME|氏名)\s*[:\s]*([A-Z][A-Z\s\-\.]+?)(?=\s*\n|国籍|$)",
            # Standalone all-caps name (at least 5 chars)
            r"(?:^|\n)\s*([A-Z][A-Z\s\-\.]{4,})(?:\s*$|\n)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Filter out common non-name matches
                exclude = ["RESIDENCE", "CARD", "JAPAN", "IMMIGRATION", "DATE", "BIRTH", 
                          "NATIONALITY", "REGION", "STATUS", "PERIOD", "STAY", "EXPIRY",
                          "WORK", "RESTRICTION", "NUMBER", "SEX"]
                if not any(ex in name for ex in exclude) and len(name) > 3:
                    return name
        return None

    def _extract_japanese_name(self, text: str) -> Optional[str]:
        """Extract Japanese name (kanji/hiragana/katakana)."""
        patterns = [
            # Japanese name pattern (kanji followed by space and more kanji/kana)
            r"(?:氏名|名前)\s*[:\s]*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+[\s　]?[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def _extract_nationality(self, text: str) -> Optional[str]:
        """Extract nationality from residence card."""
        # First check for known nationalities directly
        for jp, en in self.NATIONALITY_MAP.items():
            if jp in text:
                return en
        
        patterns = [
            r"(?:国籍)[・/\s]*(?:地域)?[:\s\n]*([^\n\d]+?)(?=\s*(?:生年|DATE|$|\n))",
            r"NATIONALITY[/\s]*REGION[:\s\n]*(.+?)(?=\s*(?:生年|DATE|$|\n))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                nationality = match.group(1).strip()
                # Check against known nationalities again
                for jp, en in self.NATIONALITY_MAP.items():
                    if jp in nationality:
                        return en
                # Filter out labels
                if nationality and nationality not in ["NATIONALITY", "REGION", "地域"]:
                    return nationality
        return None

    def _extract_region(self, text: str) -> Optional[str]:
        """Extract region (for Taiwan, Hong Kong, etc.)."""
        # Check for known nationalities as regions
        for jp, en in self.NATIONALITY_MAP.items():
            if jp in text:
                return jp  # Return Japanese version for region
        
        patterns = [
            r"地域\s*[:\s\n]*([^\n]+?)(?=\s*(?:生年|$|\n))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                region = match.group(1).strip()
                if region and region not in ["国籍", "NATIONALITY"]:
                    return region
        return None

    def _extract_residence_status(self, text: str) -> Optional[str]:
        """Extract status of residence (在留資格)."""
        # First check for known status types in text (most reliable)
        for status in self.RESIDENCE_STATUS_LIST:
            if status in text:
                return status
        
        # Then try to find after explicit label
        patterns = [
            r"(?:在留資格)\s*[:\s\n]*(.+?)(?=\s*(?:在留期間|PERIOD|$|\n))",
            r"(?:STATUS)\s*[:\s\n]*(.+?)(?=\s*(?:在留期間|PERIOD|$|\n))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                status = match.group(1).strip()
                if status and status not in ["STATUS", "在留資格"]:
                    return status
        
        return None

    def _extract_period_of_stay(self, text: str) -> Optional[str]:
        """Extract period of stay (在留期間)."""
        patterns = [
            # Pattern for Japanese format with years/months
            r"(?:在留期間)\s*[:\s\n]*(\d+年\d*月?)",
            r"(?:PERIOD\s*OF\s*STAY)\s*[:\s\n]*(\d+年\d*月?)",
            # Pattern for simple number + 年/月
            r"(\d+年(?:\d+月)?)\s*(?:まで|間)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                period = match.group(1).strip()
                if period:
                    return period
        return None

    def _extract_work_restriction(self, text: str) -> Optional[str]:
        """Extract work restriction status (就労制限)."""
        # Look for specific work restriction statuses
        restriction_statuses = [
            "就労制限なし",
            "就労不可",
            "在留資格に基づく就労活動のみ可",
            "指定書記載機関での在留資格に基づく就労活動のみ可",
            "資格外活動許可書に記載された範囲内の就労可",
        ]
        
        for status in restriction_statuses:
            if status in text:
                return status
        
        # Check for partial matches
        if "就労制限なし" in text or "制限なし" in text:
            return "就労制限なし"
        if "就労不可" in text:
            return "就労不可"
        if "資格外活動許可" in text:
            return "資格外活動許可あり"
            
        return None

    def _extract_residence_card_number(self, text: str) -> Optional[str]:
        """Extract residence card number (XX00000000XX format)."""
        match = self.residence_card_pattern.search(text)
        if match:
            return match.group(0)
        return None

    # ========== My Number Card Extraction Methods ==========

    def _extract_japanese_name_mynumber(self, text: str) -> Optional[str]:
        """Extract Japanese name from My Number Card."""
        patterns = [
            # Name on line after 氏名
            r"(?:氏名)\s*\n\s*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+[\s　]+[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)",
            r"(?:氏名)\s*[:\s]*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+[\s　]+[\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff]+)",
            r"(?:^|\n)([\u4e00-\u9faf]{1,4}[\s　]+[\u4e00-\u9faf\u3040-\u309f]+)(?=\s*\n)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Filter out non-name matches
                exclude_chars = ["生年", "月日", "有効", "番号", "住所"]
                if len(name) >= 2 and not any(ex in name for ex in exclude_chars):
                    return name
        return None

    def _extract_mynumber(self, text: str) -> Optional[str]:
        """Extract 12-digit My Number."""
        # Look for explicit label first
        patterns = [
            r"(?:個人番号|マイナンバー)\s*[:\s]*(\d{12})",
            r"(?:個人番号|マイナンバー)\s*[:\s]*(\d{4}\s*\d{4}\s*\d{4})",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                number = re.sub(r'\s', '', match.group(1))
                return number
        
        # Fall back to any 12-digit number
        match = self.mynumber_pattern.search(text)
        if match:
            return match.group(0)
        return None

    def _extract_address_mynumber(self, text: str) -> Optional[str]:
        """Extract address from My Number Card."""
        patterns = [
            r"(?:住所)\s*[:\s]*(.+?)(?=\s*(?:生年|氏名|有効|$))",
            r"([\u4e00-\u9faf]+[都道府県][\u4e00-\u9faf\d\-]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                address = match.group(1).strip()
                address = re.sub(r'\s+', ' ', address)
                if len(address) > 5:
                    return address
        return None

    # ========== Driving License Extraction Methods ==========

    def _extract_japanese_name_license(self, text: str) -> Optional[str]:
        """Extract name from driving license."""
        patterns = [
            r"(?:氏名)\s*[:\s]*([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff\s]+?)(?=\s*(?:生年|昭和|平成|令和|\d))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                if len(name) >= 2:
                    return name
        return None

    def _extract_license_number(self, text: str) -> Optional[str]:
        """Extract driving license number."""
        patterns = [
            r"(?:免許証番号|番号)\s*[:\s]*(\d[\d\-]+)",
            r"第\s*(\d{12})\s*号",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Generic 12-digit number
        match = self.license_number_pattern.search(text)
        if match:
            return match.group(0)
        return None

    def _extract_address_license(self, text: str) -> Optional[str]:
        """Extract address from driving license."""
        patterns = [
            r"(?:住所)\s*[:\s]*(.+?)(?=\s*(?:氏名|生年|交付|$|\n))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                address = match.group(1).strip()
                address = re.sub(r'\s+', ' ', address)
                if len(address) > 5:
                    return address
        return None

    def _extract_license_categories(self, text: str) -> Optional[List[str]]:
        """Extract license categories (普通, 中型, etc.)."""
        categories = []
        category_list = [
            "大型", "中型", "準中型", "普通", "大特", "大自二", "普自二", 
            "小特", "原付", "け引", "大二", "中二", "普二"
        ]
        
        for cat in category_list:
            if cat in text:
                categories.append(cat)
        
        return categories if categories else None

    def _extract_license_conditions(self, text: str) -> Optional[str]:
        """Extract license conditions (眼鏡等)."""
        # First check for common known conditions
        known_conditions = [
            "眼鏡等", "AT限定", "補聴器", "大型車(8t)に限る", 
            "中型車(8t)に限る", "準中型車(5t)に限る"
        ]
        for cond in known_conditions:
            if cond in text:
                return cond
        
        patterns = [
            r"(?:免許の条件|条件等?)[:\s]*([^\n]+?)(?=\s*(?:$|\n|種類|交付|備考))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                conditions = match.group(1).strip()
                if conditions and conditions not in ["なし", "等"]:
                    return conditions
        return None

    def _extract_issuing_authority(self, text: str) -> Optional[str]:
        """Extract issuing authority (公安委員会)."""
        patterns = [
            r"([\u4e00-\u9faf]+公安委員会)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    # ========== Common Extraction Methods ==========

    def _extract_gender(self, text: str) -> Optional[str]:
        """Extract and normalize gender."""
        patterns = [
            r"(?:性別|SEX|GENDER)\s*[:\s]*(男|女|男性|女性|M|F|Male|Female)",
            r"(?:^|\s)(男|女)(?:\s|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_gender = match.group(1).strip()
                return self.GENDER_MAP.get(raw_gender, raw_gender)
        return None

    def _extract_date_field(self, text: str, labels: List[str]) -> Optional[str]:
        """Extract date following specific labels."""
        for label in labels:
            # Try to find date after label
            pattern = rf"{re.escape(label)}\s*[:\s]*([^\n]+?)(?=\s*(?:$|\n|[a-zA-Z\u4e00-\u9faf]{{2}}))"
            match = re.search(pattern, text)
            if match:
                date_text = match.group(1).strip()
                # Validate it looks like a date
                date_match = self.jp_date_pattern.search(date_text)
                if date_match:
                    for g in date_match.groups():
                        if g:
                            return g
                # Return raw if it contains year/month/day indicators
                if re.search(r'\d+年|\d+月|\d+日|令和|平成|昭和', date_text):
                    return date_text
        
        # Fall back to finding any date near labels
        for label in labels:
            if label in text:
                idx = text.find(label)
                search_area = text[idx:idx+50]
                date_match = self.jp_date_pattern.search(search_area)
                if date_match:
                    for g in date_match.groups():
                        if g:
                            return g
        return None

    def _extract_address(self, text: str) -> Optional[str]:
        """Extract address from residence card."""
        patterns = [
            r"(?:住居地|住所|ADDRESS)\s*[:\s]*(.+?)(?=\s*(?:[A-Z]{2}\d{8}[A-Z]{2}|在留カード番号|RESIDENCE CARD NUMBER|在留|就労|$|\n\n))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                address = match.group(1).strip()
                address = re.sub(r'\s+', ' ', address)
                # Remove any card numbers that may have been captured
                address = re.sub(r'[A-Z]{2}\d{8}[A-Z]{2}', '', address).strip()
                if len(address) > 5:
                    return address
        return None

    def _empty_result(self, document_type: str) -> dict:
        """Return empty result structure for document type."""
        if document_type == self.DOC_RESIDENCE_CARD:
            return {"document_type": document_type, **{f: None for f in self.RESIDENCE_CARD_FIELDS}}
        elif document_type == self.DOC_MYNUMBER_CARD:
            return {"document_type": document_type, **{f: None for f in self.MYNUMBER_CARD_FIELDS}}
        elif document_type == self.DOC_DRIVING_LICENSE:
            return {"document_type": document_type, **{f: None for f in self.DRIVING_LICENSE_FIELDS}}
        return {"document_type": document_type}