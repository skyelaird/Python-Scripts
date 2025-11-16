"""Comprehensive name parsing utilities for genealogy data.

This module provides robust parsing for NAME, GIVN, SURN, NICK, and other name fields,
handling complex cases like:
- Ordinals and epithets in given names (Thomas II 'The Wise')
- Nobility titles as suffixes (1st Baron, Duke of York)
- Noble surname patterns (von Franconia, de France, van der Berg)
- Honorific prefixes (Frau, Herr, Sir, Lady, etc.)
- Quoted nicknames and epithets
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParsedName:
    """Represents a fully parsed name with all components."""
    given: Optional[str] = None
    surname: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    nickname: Optional[str] = None

    # Additional parsing metadata
    ordinal: Optional[str] = None  # Roman numerals like II, III
    epithets: List[str] = None  # Quoted epithets like 'The Wise'

    def __post_init__(self):
        if self.epithets is None:
            self.epithets = []


class NameParser:
    """Parser for complex genealogical name fields."""

    # Honorific prefixes by language
    PREFIXES = {
        'English': ['Sir', 'Lady', 'Lord', 'Dame', 'Mr', 'Mrs', 'Miss', 'Ms',
                   'Dr', 'Rev', 'Revd', 'Father', 'Mother'],
        'French': ['M', 'Mme', 'Mlle', 'Sieur', 'Dame', 'Seigneur', 'Messire'],
        'German': ['Herr', 'Frau', 'Fräulein', 'Freiherr', 'Freiin'],
        'Spanish': ['Sr', 'Sra', 'Srta', 'Don', 'Doña', 'Señor', 'Señora'],
        'Italian': ['Sig', 'Sig.ra', 'Sig.na', 'Signore', 'Signora', 'Donna'],
        'Dutch': ['Heer', 'Mevrouw', 'Juffrouw'],
        'Portuguese': ['Sr', 'Sra', 'Srta', 'Senhor', 'Senhora', 'Dom', 'Dona']
    }

    # Flatten prefix list for easy matching
    ALL_PREFIXES = []
    for prefixes in PREFIXES.values():
        ALL_PREFIXES.extend(prefixes)

    # Noble surname particles that indicate surnames, NOT nicknames
    SURNAME_PARTICLES = [
        # German
        'von', 'vom', 'zu', 'zur', 'zum', 'im', 'am',
        # French
        'de', 'du', 'des', 'd\'', 'de la', 'de l\'', 'du',
        # Dutch
        'van', 'van de', 'van den', 'van der', 'van het', 'ter', 'te', 'ten',
        # Italian
        'di', 'da', 'del', 'della', 'degli', 'delle',
        # Spanish/Portuguese
        'del', 'de la', 'de los', 'de las', 'dos', 'das',
        # Irish/Scottish
        'mac', 'mc', 'o\'',
        # Other
        'af', 'av', 'von und zu'
    ]

    # Ordinal patterns (Roman numerals)
    ORDINAL_PATTERN = re.compile(r'\b([IVX]+)\b$')

    # Quoted epithet patterns (nicknames in quotes)
    QUOTED_EPITHET_PATTERN = re.compile(r'''['"]([^'"]+)['"]''')

    # Nobility suffix patterns
    NOBILITY_SUFFIX_PATTERNS = [
        # English ordinal titles
        r'\b(\d+(?:st|nd|rd|th)\s+(?:Earl|Baron|Baroness|Lord|Duke|Duchess|Count|Countess)(?:\s+[Oo]f\s+\w+)?)\b',
        # Other English titles
        r'\b(Earl|Baron|Baroness|Lord|Duke|Duchess|Count|Countess|Marquess|Marchioness|Viscount|Viscountess)(?:\s+[Oo]f\s+\w+)?\b',
        # Heiress/Heir of X
        r'\b((?:Heir|Heiress)\s+[Oo]f\s+\w+)\b',
        # German titles
        r'\b(Graf|Herzog|Fürst|Baron)(?:\s+(?:von|im|zu)\s+\w+)?\b',
        # French titles
        r'\b(Comte|Duc|Marquis|Baron)(?:\s+d[e\']\s*\w+)?\b',
    ]

    # Compile nobility patterns
    COMPILED_NOBILITY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in NOBILITY_SUFFIX_PATTERNS]

    @classmethod
    def is_surname_particle(cls, word: str) -> bool:
        """Check if a word is a noble surname particle (von, de, van, etc.)."""
        word_lower = word.lower()
        # Remove trailing apostrophes for matching
        word_normalized = word_lower.rstrip("'")
        return word_normalized in cls.SURNAME_PARTICLES or word_lower in cls.SURNAME_PARTICLES

    @classmethod
    def is_prefix(cls, word: str) -> bool:
        """Check if a word is an honorific prefix."""
        word_clean = word.rstrip('.')
        return word_clean in cls.ALL_PREFIXES or word_clean.lower().capitalize() in cls.ALL_PREFIXES

    @classmethod
    def extract_quoted_epithets(cls, text: str) -> Tuple[str, List[str]]:
        """Extract quoted epithets from text and return cleaned text and list of epithets.

        Args:
            text: Text potentially containing quoted epithets

        Returns:
            Tuple of (cleaned_text, list_of_epithets)
        """
        epithets = cls.QUOTED_EPITHET_PATTERN.findall(text)
        cleaned = cls.QUOTED_EPITHET_PATTERN.sub('', text).strip()
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned, epithets

    @classmethod
    def extract_ordinal(cls, text: str) -> Tuple[str, Optional[str]]:
        """Extract Roman numeral ordinal from end of text.

        Args:
            text: Text potentially ending with ordinal (e.g., "Thomas II")

        Returns:
            Tuple of (text_without_ordinal, ordinal)
        """
        match = cls.ORDINAL_PATTERN.search(text)
        if match:
            ordinal = match.group(1)
            # Remove ordinal from text
            text_without = text[:match.start()].strip()
            return text_without, ordinal
        return text, None

    @classmethod
    def extract_nobility_suffix(cls, text: str) -> Tuple[str, Optional[str]]:
        """Extract nobility title suffix from text.

        Args:
            text: Text potentially containing nobility suffix

        Returns:
            Tuple of (text_without_suffix, suffix)
        """
        for pattern in cls.COMPILED_NOBILITY_PATTERNS:
            match = pattern.search(text)
            if match:
                suffix = match.group(0)
                # Remove suffix from text
                text_without = text[:match.start()] + text[match.end():]
                text_without = re.sub(r'\s+', ' ', text_without).strip()
                return text_without, suffix
        return text, None

    @classmethod
    def extract_prefix(cls, text: str) -> Tuple[str, Optional[str]]:
        """Extract honorific prefix from beginning of text.

        Args:
            text: Text potentially starting with prefix

        Returns:
            Tuple of (text_without_prefix, prefix)
        """
        words = text.split()
        if not words:
            return text, None

        first_word = words[0].rstrip('.')
        if cls.is_prefix(first_word):
            prefix = words[0]  # Keep original with punctuation
            remaining = ' '.join(words[1:])
            return remaining, prefix

        return text, None

    @classmethod
    def has_surname_particle(cls, text: str) -> bool:
        """Check if text contains a surname particle anywhere."""
        if not text:
            return False
        words = text.split()
        return any(cls.is_surname_particle(word) for word in words)

    @classmethod
    def extract_surname_with_particle(cls, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract surname from text that may contain surname particles.

        This is useful for parsing fields that mix given names and surnames.

        Args:
            text: Text potentially containing "Given von Surname" pattern

        Returns:
            Tuple of (remaining_text, surname_with_particle)
        """
        if not text:
            return None, None

        words = text.split()

        # Find first surname particle
        for i, word in enumerate(words):
            if cls.is_surname_particle(word):
                # Everything from this particle onward is the surname
                surname = ' '.join(words[i:])
                remaining = ' '.join(words[:i]).strip() if i > 0 else None
                return remaining, surname

        return text, None

    @classmethod
    def parse_givn_field(cls, givn_value: str) -> ParsedName:
        """Parse a GIVN field that may contain ordinals, epithets, and titles.

        Example inputs:
            "Thomas II 'The Wise' 1st Baron" ->
                given: "Thomas II"
                nickname: "The Wise"
                suffix: "1st Baron"
                ordinal: "II"

        Args:
            givn_value: Value from GIVN field

        Returns:
            ParsedName object with extracted components
        """
        result = ParsedName()
        text = givn_value.strip()

        if not text:
            return result

        # Step 1: Extract quoted epithets (these become nicknames)
        text, epithets = cls.extract_quoted_epithets(text)
        result.epithets = epithets
        if epithets:
            result.nickname = epithets[0]  # Use first epithet as primary nickname

        # Step 2: Extract nobility suffix
        text, suffix = cls.extract_nobility_suffix(text)
        if suffix:
            result.suffix = suffix

        # Step 3: Extract ordinal (if present, keep it with given name)
        text_without_ordinal, ordinal = cls.extract_ordinal(text)
        if ordinal:
            result.ordinal = ordinal
            result.given = text  # Keep ordinal with given name "Thomas II"
        else:
            result.given = text

        return result

    @classmethod
    def parse_name_field(cls, name_value: str) -> ParsedName:
        """Parse a full NAME field in GEDCOM format.

        GEDCOM NAME format: "Given /Surname/ Suffix"
        But may also contain prefixes, epithets, etc.

        Example inputs:
            "Thomas II /Smith/" -> given: "Thomas II", surname: "Smith"
            "Sir Thomas /Smith/ 1st Baron" -> prefix: "Sir", given: "Thomas",
                                               surname: "Smith", suffix: "1st Baron"
            "Frau Gerberga /von Franconia/" -> prefix: "Frau", given: "Gerberga",
                                                surname: "von Franconia"

        Args:
            name_value: Value from NAME field

        Returns:
            ParsedName object with extracted components
        """
        result = ParsedName()
        text = name_value.strip()

        if not text:
            return result

        # Extract surname from /Surname/ format
        surname_match = re.search(r'/([^/]*)/', text)
        if surname_match:
            result.surname = surname_match.group(1).strip()
            # Remove surname from text
            text = text[:surname_match.start()] + text[surname_match.end():]
            text = text.strip()

        # What remains is given name and possibly prefix/suffix
        if not text:
            return result

        # Extract prefix from beginning
        text, prefix = cls.extract_prefix(text)
        if prefix:
            result.prefix = prefix

        # Extract quoted epithets
        text, epithets = cls.extract_quoted_epithets(text)
        result.epithets = epithets
        if epithets:
            result.nickname = epithets[0]

        # Extract nobility suffix
        text, suffix = cls.extract_nobility_suffix(text)
        if suffix:
            result.suffix = suffix

        # What remains is the given name (possibly with ordinal)
        text_without_ordinal, ordinal = cls.extract_ordinal(text)
        if ordinal:
            result.ordinal = ordinal
            result.given = text  # Keep ordinal with given name
        else:
            result.given = text if text else None

        return result

    @classmethod
    def parse_field_with_surname_particle(cls, field_value: str, field_type: str = 'UNKNOWN') -> ParsedName:
        """Parse a field that incorrectly contains surname particles.

        This handles cases where a field tagged as NICK or GIVN actually contains
        a surname with particles like "von Franconia" or "de France".

        Example:
            "Frau Gerberga von Franconia" (tagged as NICK) ->
                prefix: "Frau"
                given: "Gerberga"
                surname: "von Franconia"

        Args:
            field_value: The field value
            field_type: Type hint (NICK, GIVN, etc.) for better parsing

        Returns:
            ParsedName object with corrected components
        """
        result = ParsedName()
        text = field_value.strip()

        if not text:
            return result

        # Extract prefix first
        text, prefix = cls.extract_prefix(text)
        if prefix:
            result.prefix = prefix

        # Extract quoted epithets
        text, epithets = cls.extract_quoted_epithets(text)
        result.epithets = epithets

        # Check for surname particle
        remaining, surname = cls.extract_surname_with_particle(text)
        if surname:
            result.surname = surname
            if remaining:
                result.given = remaining
        else:
            # No surname particle found, treat as given name
            result.given = text if text else None

        return result

    @classmethod
    def normalize_name_components(cls, given: Optional[str] = None,
                                  surname: Optional[str] = None,
                                  prefix: Optional[str] = None,
                                  suffix: Optional[str] = None,
                                  nickname: Optional[str] = None) -> ParsedName:
        """Normalize and validate name components, applying parsing rules.

        This method takes individual name components and applies parsing rules to
        ensure they are in the correct fields. For example, if 'given' contains
        surname particles, it will extract them.

        Args:
            given: Given name value
            surname: Surname value
            prefix: Prefix value
            suffix: Suffix value
            nickname: Nickname value

        Returns:
            ParsedName object with normalized components
        """
        result = ParsedName()

        # Process given name for embedded components
        if given:
            given_parsed = cls.parse_givn_field(given)
            result.given = given_parsed.given
            result.ordinal = given_parsed.ordinal
            if not nickname and given_parsed.nickname:
                result.nickname = given_parsed.nickname
            if not suffix and given_parsed.suffix:
                result.suffix = given_parsed.suffix

        # Process nickname for embedded surname particles
        if nickname and cls.has_surname_particle(nickname):
            nick_parsed = cls.parse_field_with_surname_particle(nickname, 'NICK')
            # If it contains surname particles, it's misclassified
            if nick_parsed.surname and not surname:
                result.surname = nick_parsed.surname
                if nick_parsed.given and not result.given:
                    result.given = nick_parsed.given
                if nick_parsed.prefix and not prefix:
                    result.prefix = nick_parsed.prefix
                # Don't use this as a nickname
                nickname = None

        # Set remaining components
        result.surname = result.surname or surname
        result.prefix = result.prefix or prefix
        result.suffix = result.suffix or suffix
        result.nickname = result.nickname or nickname

        return result
