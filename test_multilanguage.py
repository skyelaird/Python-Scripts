#!/usr/bin/env python3
"""Test script for multi-language placeholder detection."""

import sys
import unicodedata

# Inline the normalize function for testing
def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for language-agnostic comparison.

    This removes accents and converts to uppercase for comparison,
    so "Señora" matches "SENORA", "Madame" matches "MADAME", etc.
    """
    if not text:
        return ''

    # Normalize unicode to decomposed form (NFD)
    nfd = unicodedata.normalize('NFD', text)

    # Remove accent marks (combining characters)
    without_accents = ''.join(
        char for char in nfd
        if unicodedata.category(char) != 'Mn'
    )

    return without_accents.upper().strip()


# Inline the placeholder titles for testing
PLACEHOLDER_TITLES = {
    # English
    'MRS', 'MRS.', 'MS', 'MS.', 'MISS', 'MR', 'MR.',

    # French
    'MME', 'MME.', 'MADAME', 'M', 'M.', 'MONSIEUR', 'MLLE', 'MLLE.', 'MADEMOISELLE',

    # Spanish
    'SRA', 'SRA.', 'SEÑORA', 'SENORA', 'SR', 'SR.', 'SEÑOR', 'SENOR',
    'SRTA', 'SRTA.', 'SEÑORITA', 'SRITA', 'SRITA.',

    # Portuguese
    'SENHORA', 'SENHOR',

    # German
    'FRAU', 'HERR',

    # Dutch
    'MEVROUW', 'MEVR', 'MEVR.', 'MIJNHEER', 'DHR', 'DHR.',

    # Italian
    'SIG.RA', 'SIGNORA', 'SIG', 'SIG.', 'SIGNORE', 'SIGNORINA',

    # Catalan
    'SRA', 'SR', 'SENYORA', 'SENYOR',

    # Other common patterns
    'DAME', 'LADY', 'LORD', 'SIR',
}

PLACEHOLDER_PREFIXES = {
    'MRS.', 'MRS ', 'MS.', 'MS ', 'MISS ', 'MR.', 'MR ',
    'MME.', 'MME ', 'M.', 'M ', 'MLLE.', 'MLLE ',
    'SRA.', 'SRA ', 'SR.', 'SR ', 'SRTA.', 'SRTA ',
    'FRAU ', 'HERR ',
    'MEVROUW ', 'MEVR.', 'MEVR ', 'MIJNHEER ', 'DHR.', 'DHR ',
    'SIG.RA ', 'SIG.', 'SIG ',
}


def test_normalization():
    """Test Unicode normalization for various languages."""
    print("Testing Unicode normalization...")
    print("="*60)

    test_cases = [
        ("Señora", "SENORA"),
        ("señora", "SENORA"),
        ("SEÑORA", "SENORA"),
        ("Madame", "MADAME"),
        ("Mme", "MME"),
        ("Mme.", "MME."),
        ("Frau", "FRAU"),
        ("Müller", "MULLER"),
        ("François", "FRANCOIS"),
        ("José", "JOSE"),
        ("Renée", "RENEE"),
        ("Château", "CHATEAU"),
    ]

    all_passed = True
    for input_text, expected in test_cases:
        result = normalize_for_comparison(input_text)
        passed = result == expected
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"{status} '{input_text}' -> '{result}' (expected '{expected}')")

    print()
    return all_passed


def test_placeholder_detection():
    """Test placeholder title detection."""
    print("Testing placeholder title detection...")
    print("="*60)

    # Test cases: (given_name, should_match)
    test_cases = [
        # English
        ("Mrs", True),
        ("Mrs.", True),
        ("MS", True),
        ("Miss", True),
        ("Mr", True),

        # French
        ("Mme", True),
        ("Mme.", True),
        ("Madame", True),
        ("M.", True),
        ("Monsieur", True),

        # Spanish
        ("Sra", True),
        ("Señora", True),
        ("Sra.", True),
        ("Sr", True),
        ("Señor", True),

        # German
        ("Frau", True),
        ("Herr", True),

        # Dutch
        ("Mevrouw", True),
        ("Mijnheer", True),

        # Italian
        ("Signora", True),
        ("Sig.ra", True),

        # Real names (should NOT match)
        ("Marie", False),
        ("Jean", False),
        ("Maria", False),
        ("François", False),
        ("José", False),
    ]

    all_passed = True
    for given_name, should_match in test_cases:
        normalized = normalize_for_comparison(given_name)
        is_match = normalized in PLACEHOLDER_TITLES
        passed = is_match == should_match
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        match_text = "placeholder" if is_match else "real name"
        print(f"{status} '{given_name}' -> {match_text} (expected: {'placeholder' if should_match else 'real name'})")

    print()
    return all_passed


def test_prefix_detection():
    """Test placeholder prefix detection (e.g., 'Mrs. John Smith')."""
    print("Testing placeholder prefix detection...")
    print("="*60)

    # Test cases: (given_name, should_have_prefix)
    test_cases = [
        ("Mrs. John Smith", True),
        ("Mme. Jean Dupont", True),
        ("Sra. Maria Garcia", True),
        ("Frau Hans Mueller", True),
        ("Marie Curie", False),
        ("Jean-Paul Sartre", False),
    ]

    all_passed = True
    for given_name, should_have_prefix in test_cases:
        normalized = normalize_for_comparison(given_name)
        has_prefix = any(normalized.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)
        passed = has_prefix == should_have_prefix
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        prefix_text = "has prefix" if has_prefix else "no prefix"
        print(f"{status} '{given_name}' -> {prefix_text} (expected: {'has prefix' if should_have_prefix else 'no prefix'})")

    print()
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MULTI-LANGUAGE SUPPORT TEST SUITE")
    print("="*60)
    print()

    test_results = []

    test_results.append(("Unicode Normalization", test_normalization()))
    test_results.append(("Placeholder Detection", test_placeholder_detection()))
    test_results.append(("Prefix Detection", test_prefix_detection()))

    print("="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in test_results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
        all_passed = all_passed and passed

    print()
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
