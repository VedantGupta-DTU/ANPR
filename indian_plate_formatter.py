"""
Indian License Plate Formatter
Validates and formats OCR output to match Indian license plate standards

Indian License Plate Format:
- State Code: 2 letters (e.g., MH, DL, KA, HR, PB, UP, TN, GJ, RJ)
- District Code: 2 digits (01-99)
- Series: 1-3 letters (A, AB, ABC)
- Number: 1-4 digits (1-9999)

Examples:
- MH 12 AB 1234
- DL 7C Q 1939
- KA 51 N 0099
"""
import re
from typing import Optional, Tuple

# Valid Indian state codes
INDIAN_STATE_CODES = [
    "AN",  # Andaman and Nicobar Islands
    "AP",  # Andhra Pradesh
    "AR",  # Arunachal Pradesh
    "AS",  # Assam
    "BR",  # Bihar
    "CH",  # Chandigarh
    "CG",  # Chhattisgarh
    "DD",  # Daman and Diu
    "DL",  # Delhi
    "GA",  # Goa
    "GJ",  # Gujarat
    "HP",  # Himachal Pradesh
    "HR",  # Haryana
    "JH",  # Jharkhand
    "JK",  # Jammu and Kashmir
    "KA",  # Karnataka
    "KL",  # Kerala
    "LA",  # Ladakh
    "LD",  # Lakshadweep
    "MH",  # Maharashtra
    "ML",  # Meghalaya
    "MN",  # Manipur
    "MP",  # Madhya Pradesh
    "MZ",  # Mizoram
    "NL",  # Nagaland
    "OD",  # Odisha (also OR)
    "OR",  # Odisha (old code)
    "PB",  # Punjab
    "PY",  # Puducherry
    "RJ",  # Rajasthan
    "SK",  # Sikkim
    "TN",  # Tamil Nadu
    "TS",  # Telangana
    "TR",  # Tripura
    "UK",  # Uttarakhand (also UA)
    "UA",  # Uttarakhand (old code)
    "UP",  # Uttar Pradesh
    "WB",  # West Bengal
]

# Common OCR misreadings and corrections
OCR_CORRECTIONS = {
    # Numbers misread as letters
    '0': 'O',  # Can be either
    'O': '0',  # Can be either
    '1': 'I',  # Can be either
    'I': '1',  # Can be either
    '5': 'S',
    'S': '5',
    '8': 'B',
    'B': '8',
    '6': 'G',
    'G': '6',
    '2': 'Z',
    'Z': '2',
    # Special characters to remove
    '@': 'A',
    '#': 'H',
    '$': 'S',
    '&': '8',
}


class IndianPlateFormatter:
    """Formats and validates Indian license plate numbers"""
    
    def __init__(self):
        self.state_codes = set(INDIAN_STATE_CODES)
        
        # Pattern for Indian license plate
        # Format: SS DD SSS NNNN (State-District-Series-Number)
        # The series can be 1-3 letters, number can be 1-4 digits
        self.plate_pattern = re.compile(
            r'^([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{1,3})\s*(\d{1,4})$'
        )
        
        # Alternative pattern for BH (Bharat) series plates
        self.bh_pattern = re.compile(
            r'^(\d{2})\s*BH\s*(\d{4})\s*([A-Z]{1,2})$'
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean OCR output text
        
        Args:
            text: Raw OCR output
            
        Returns:
            Cleaned text with only alphanumeric characters
        """
        # Convert to uppercase
        text = text.upper()
        
        # Collapse newlines and multi-line OCR output into a single line
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove common prefixes that might be detected
        prefixes_to_remove = ['IND', 'INDIA', 'AND', 'NON']
        for prefix in prefixes_to_remove:
            if text.startswith(prefix + ' '):
                text = text[len(prefix):].strip()
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Also remove these prefixes if they appear after the plate text (trailing junk)
        for suffix in ['NON', 'IND', 'INDIA']:
            if text.endswith(' ' + suffix):
                text = text[:-(len(suffix) + 1)].strip()
        
        # Replace common OCR errors
        for wrong, right in [('@', 'A'), ('#', 'H'), ('$', 'S'), ('/', ''), ('\\', ''), ('|', '1'), ('~', ''), ('`', '')]:
            text = text.replace(wrong, right)
        
        # Remove all non-alphanumeric characters except spaces
        text = re.sub(r'[^A-Z0-9\s]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_components(self, text: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Extract license plate components from text
        
        Args:
            text: Cleaned text
            
        Returns:
            Tuple of (state_code, district_code, series, number) or None
        """
        text = self.clean_text(text)
        
        # Remove all spaces for pattern matching
        compact = text.replace(' ', '')
        
        # Heuristic: if raw text is already valid, we should still try corrections
        # because OCR might read digits as letters (e.g. G -> 6) which fits regex but is wrong.
        # So we calculate corrected version first.
        
        corrected = self._apply_ocr_corrections(compact)
        
        # Check corrected version
        match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', corrected)
        if match:
            state, district, series, number = match.groups()
            
            # Try alt corrections (0->Q instead of 0->O) to see if it produces
            # a better result — both must be valid, pick the one with valid state
            alt_corrected = self._apply_ocr_corrections_alt(compact)
            alt_match = re.match(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', alt_corrected)
            
            if alt_match:
                alt_state, alt_dist, alt_series, alt_num = alt_match.groups()
                # Prefer alt only if it changes the series (Q vs O) and
                # the alt state is valid (or both states are equally valid)
                if alt_series != series:
                    primary_state_valid = state in self.state_codes
                    alt_state_valid = alt_state in self.state_codes
                    if alt_state_valid and not primary_state_valid:
                        return alt_match.groups()
            
            return match.groups()
             
        # If corrected verification failed, check if raw was valid
        match = re.search(r'^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$', compact)
        if match:
             return match.groups()

        # Try OCR-corrected version with search (not anchored) - fallback
        match = re.search(r'([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})', corrected)
        if match:
             return match.groups()
        
        return None
    
    def _apply_ocr_corrections_alt(self, text: str) -> str:
        """
        Alternative OCR corrections: tries O instead of Q for series positions,
        and 6 instead of 0 for G in number positions. Used as a fallback when
        the primary corrections produce an invalid state code.
        """
        if len(text) < 6:
            return text
        result = list(text)
        # State: same as primary
        digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z', '7': 'T'}
        for i in range(min(2, len(result))):
            if result[i] in digit_to_letter:
                result[i] = digit_to_letter[result[i]]
        # District: same as primary
        letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'G': '0', 'Z': '2', 'T': '1', 'Q': '0'}
        for i in range(2, min(4, len(result))):
            if result[i] in letter_to_digit:
                result[i] = letter_to_digit[result[i]]
        if len(result) > 2 and result[2].isdigit():
            if len(result) > 3 and (result[3].isdigit() or result[3] in letter_to_digit):
                dist_end = 4
            else:
                dist_end = 3
        else:
            dist_end = 2
        for i in range(2, min(dist_end, len(result))):
            if result[i] in letter_to_digit:
                result[i] = letter_to_digit[result[i]]
        num_start = len(result)
        for i in range(len(result) - 1, dist_end - 1, -1):
            if result[i].isdigit() or result[i] in letter_to_digit:
                num_start = i
            else:
                break
        num_len = len(result) - num_start
        if num_len > 4:
            num_start += (num_len - 4)
        # Series: use O instead of Q (alternative)
        alt_series_d2l = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z', '7': 'T'}
        for i in range(dist_end, min(num_start, len(result))):
            if result[i] in alt_series_d2l:
                result[i] = alt_series_d2l[result[i]]
        # Number: same as primary
        number_l2d = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'G': '0', 'Z': '2', 'T': '1', 'Q': '0'}
        for i in range(num_start, len(result)):
            if result[i] in number_l2d:
                result[i] = number_l2d[result[i]]
        return ''.join(result)
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """
        Apply context-aware OCR corrections
        
        For Indian plates:
        - First 2 chars should be letters (state code)
        - Next 1-2 chars should be digits (district)
        - Next 1-3 chars should be letters (series)
        - Last 1-4 chars should be digits (number)
        """
        if len(text) < 6:
            return text
        
        result = list(text)
        
        # --- Position 0-1: State code (must be letters) ---
        digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z', '7': 'T'}
        for i in range(min(2, len(result))):
            if result[i] in digit_to_letter:
                result[i] = digit_to_letter[result[i]]
        
        # --- Position 2-3: District code (must be digits) ---
        # G -> 0 here (on plates, G is almost always a misread 0, not 6)
        letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'G': '0', 'Z': '2', 'T': '1', 'Q': '0'}
        for i in range(2, min(4, len(result))):
            if result[i] in letter_to_digit:
                result[i] = letter_to_digit[result[i]]
        
        # --- Find where series (letters) starts and trailing number (digits) begins ---
        # After the district code (pos 2-3), find consecutive letters = series,
        # then remaining chars = number (digits)
        # Determine district length: check if char at position 2 and 3 are digits
        if len(result) > 2 and result[2].isdigit():
            # At least one digit at pos 2
            if len(result) > 3 and (result[3].isdigit() or result[3] in letter_to_digit):
                 dist_end = 4
            else:
                 dist_end = 3
        else:
            # Fallback (maybe district matches letter_to_digit mapping that wasn't applied?)
            dist_end = 2
        
        # Ensure district positions are digits
        for i in range(2, min(dist_end, len(result))):
            if result[i] in letter_to_digit:
                result[i] = letter_to_digit[result[i]]
        
        # --- Find the trailing digits (number) by scanning from the end ---
        # Work backwards to find where the final number block starts
        num_start = len(result)
        for i in range(len(result) - 1, dist_end - 1, -1):
            is_dig = result[i].isdigit()
            in_map = result[i] in letter_to_digit
            if is_dig or in_map:
                num_start = i
            else:
                break
        
        # Heuristic: Indian plates always have exactly 4 trailing digits as the number.
        # If the number block has > 4 digits, the extra leading digits are actually
        # series letters misread as digits by OCR (e.g. Q -> 0, S -> 5).
        # Move those extra digits into the series region so they get letter-converted.
        num_len = len(result) - num_start
        if num_len > 4:
            extra_digits = num_len - 4
            num_start += extra_digits  # Shift number start to keep only last 4 digits
            # The region [dist_end : num_start] is now Series, and will be letter-converted below

        
        # --- Series region: chars between district and number (should be letters) ---
        # Default: 0 -> O (most common). The Q variant is tried separately via
        # _apply_ocr_corrections_alt and extract_components picks the best match.
        series_digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z', '7': 'T'}
        for i in range(dist_end, min(num_start, len(result))):
            if result[i] in series_digit_to_letter:
                result[i] = series_digit_to_letter[result[i]]
        
        # --- Number region: trailing chars (should be digits) ---
        # G -> 0 (G is almost always a misread 0 on plates, not 6)
        number_letter_to_digit = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'G': '0', 'Z': '2', 'T': '1', 'Q': '0'}
        for i in range(num_start, len(result)):
            if result[i] in number_letter_to_digit:
                result[i] = number_letter_to_digit[result[i]]
        
        return ''.join(result)
    
    def format_plate(self, text: str) -> str:
        """
        Format text as a standard Indian license plate
        
        Args:
            text: OCR output text
            
        Returns:
            Formatted plate number (e.g., "MH 12 AB 1234")
        """
        components = self.extract_components(text)
        
        if components:
            state, district, series, number = components
            
            # Validate state code
            if state not in self.state_codes and state != 'BH':
                # Try to find closest match
                state = self._find_closest_state(state)
            
            # Keep district and number as-is (no zero padding)
            
            return f"{state} {district} {series} {number}"
        
        # If pattern matching failed, just clean and return
        return self.clean_text(text)
    
    def _find_closest_state(self, code: str) -> str:
        """Find the closest matching state code"""
        if len(code) != 2:
            return code
        
        # Check for common OCR errors in state codes
        corrections = {
            'NH': 'MH',  # Maharashtra
            'OK': 'DL',  # Delhi
            'KK': 'KA',  # Karnataka
            'RI': 'RJ',  # Rajasthan
            'OL': 'DL',  # Delhi
        }
        
        if code in corrections:
            return corrections[code]
        
        return code
    
    def validate_plate(self, text: str) -> Tuple[bool, str]:
        """
        Validate if text is a valid Indian license plate
        
        Args:
            text: Plate text to validate
            
        Returns:
            Tuple of (is_valid, formatted_plate)
        """
        components = self.extract_components(text)
        
        if not components:
            return False, self.clean_text(text)
        
        state, district, series, number = components
        
        # Validate state code
        is_valid_state = state in self.state_codes or state == 'BH'
        
        # Validate district (1-99)
        try:
            dist_num = int(district)
            is_valid_district = 1 <= dist_num <= 99
        except ValueError:
            is_valid_district = False
        
        # Validate number (1-9999)
        try:
            num = int(number)
            is_valid_number = 1 <= num <= 9999
        except ValueError:
            is_valid_number = False
        
        is_valid = is_valid_state and is_valid_district and is_valid_number
        formatted = self.format_plate(text)
        
        return is_valid, formatted


def format_indian_plate(text: str) -> str:
    """
    Convenience function to format an Indian license plate
    
    Args:
        text: Raw OCR output
        
    Returns:
        Formatted Indian license plate
    """
    formatter = IndianPlateFormatter()
    return formatter.format_plate(text)


def validate_indian_plate(text: str) -> Tuple[bool, str]:
    """
    Convenience function to validate an Indian license plate
    
    Args:
        text: Raw OCR output
        
    Returns:
        Tuple of (is_valid, formatted_plate)
    """
    formatter = IndianPlateFormatter()
    return formatter.validate_plate(text)


# Test the formatter
if __name__ == "__main__":
    formatter = IndianPlateFormatter()
    
    test_cases = [
        "IND MH 12 AB 3456",
        "AND DL 7CQ 1939",
        "MH43CC1745/",
        "KA 64 N 0099",
        "MH20 DV2363",
        "PB46 DZ687",
        "MHG7AG4423",
        "WP53@VGOOD",
        "#R123C0547 1",
    ]
    
    print("Indian License Plate Formatter Test")
    print("=" * 50)
    
    for test in test_cases:
        is_valid, formatted = formatter.validate_plate(test)
        status = "✓" if is_valid else "✗"
        print(f"{status} '{test}' -> '{formatted}'")
