"""
Reference data loader for noble titles and place name variants.

This module loads and normalizes multilingual reference data for genealogical matching.
It supports noble titles, professional suffixes, generational suffixes, and place names
across multiple languages.
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TitleVariant:
    """Represents a noble title with multilingual variants."""
    canonical: str
    rank: Optional[int] = None
    category: Optional[str] = None
    variants: Dict[str, List[str]] = field(default_factory=dict)
    feminine: Dict[str, List[str]] = field(default_factory=dict)

    def get_all_variants(self, include_feminine: bool = True) -> Set[str]:
        """Get all variants across all languages (normalized to lowercase)."""
        all_variants = set()
        for lang_variants in self.variants.values():
            all_variants.update(v.lower() for v in lang_variants)

        if include_feminine:
            for lang_variants in self.feminine.values():
                all_variants.update(v.lower() for v in lang_variants)

        return all_variants

    def get_variants_for_language(self, language: str, include_feminine: bool = True) -> List[str]:
        """Get variants for a specific language."""
        variants = self.variants.get(language, []).copy()
        if include_feminine and language in self.feminine:
            variants.extend(self.feminine[language])
        return [v.lower() for v in variants]

    def normalize(self, suffix: str) -> Optional[str]:
        """Normalize a suffix variant to canonical form."""
        suffix_lower = suffix.lower().strip()
        if suffix_lower in self.get_all_variants():
            return self.canonical
        return None


@dataclass
class PlaceVariant:
    """Represents a place with multilingual name variants."""
    canonical: str
    country: str
    place_type: str  # 'city', 'region', 'country'
    variants: Dict[str, List[str]] = field(default_factory=dict)

    def get_all_variants(self) -> Set[str]:
        """Get all variants across all languages (normalized to lowercase)."""
        all_variants = set()
        for lang_variants in self.variants.values():
            all_variants.update(v.lower() for v in lang_variants)
        return all_variants

    def get_variants_for_language(self, language: str) -> List[str]:
        """Get variants for a specific language."""
        return [v.lower() for v in self.variants.get(language, [])]

    def normalize(self, place_name: str) -> Optional[str]:
        """Normalize a place name variant to canonical form."""
        place_lower = place_name.lower().strip()
        if place_lower in self.get_all_variants():
            return self.canonical
        return None


class ReferenceDataLoader:
    """Loads and manages multilingual reference data for genealogical matching."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure only one instance loads the data."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the reference data loader."""
        if not ReferenceDataLoader._initialized:
            self.data_dir = Path(__file__).parent
            self.titles: List[TitleVariant] = []
            self.professional_suffixes: List[TitleVariant] = []
            self.generational_suffixes: List[TitleVariant] = []
            self.place_variants: List[PlaceVariant] = []

            # Lookup caches for fast normalization
            self._title_lookup: Dict[str, str] = {}
            self._place_lookup: Dict[str, str] = {}

            self._load_all_data()
            ReferenceDataLoader._initialized = True

    def _load_all_data(self):
        """Load all reference data files."""
        self._load_noble_titles()
        self._load_place_variants()

    def _load_noble_titles(self):
        """Load noble titles and suffixes from JSON."""
        titles_file = self.data_dir / 'noble_titles.json'

        if not titles_file.exists():
            # Gracefully handle missing file
            return

        with open(titles_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load noble titles
        for title_data in data.get('titles', []):
            title = TitleVariant(
                canonical=title_data['canonical'],
                rank=title_data.get('rank'),
                category=title_data.get('category'),
                variants=title_data.get('variants', {}),
                feminine=title_data.get('feminine', {})
            )
            self.titles.append(title)

            # Build lookup cache
            for variant in title.get_all_variants():
                self._title_lookup[variant] = title.canonical

        # Load professional suffixes
        for suffix_data in data.get('professional_suffixes', []):
            suffix = TitleVariant(
                canonical=suffix_data['canonical'],
                category=suffix_data.get('category'),
                variants=suffix_data.get('variants', {})
            )
            self.professional_suffixes.append(suffix)

            # Build lookup cache
            for variant in suffix.get_all_variants():
                self._title_lookup[variant] = suffix.canonical

        # Load generational suffixes
        for gen_data in data.get('generational_suffixes', []):
            gen_suffix = TitleVariant(
                canonical=gen_data['canonical'],
                category='generational',
                variants=gen_data.get('variants', {})
            )
            self.generational_suffixes.append(gen_suffix)

            # Build lookup cache
            for variant in gen_suffix.get_all_variants():
                self._title_lookup[variant] = gen_suffix.canonical

    def _load_place_variants(self):
        """Load place name variants from JSON."""
        places_file = self.data_dir / 'place_name_variants.json'

        if not places_file.exists():
            # Gracefully handle missing file
            return

        with open(places_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load cities
        for city_data in data.get('cities', []):
            place = PlaceVariant(
                canonical=city_data['canonical'],
                country=city_data.get('country', ''),
                place_type='city',
                variants=city_data.get('variants', {})
            )
            self.place_variants.append(place)

            # Build lookup cache
            for variant in place.get_all_variants():
                self._place_lookup[variant] = place.canonical

        # Load regions
        for region_data in data.get('regions', []):
            place = PlaceVariant(
                canonical=region_data['canonical'],
                country=region_data.get('country', ''),
                place_type='region',
                variants=region_data.get('variants', {})
            )
            self.place_variants.append(place)

            # Build lookup cache
            for variant in place.get_all_variants():
                self._place_lookup[variant] = place.canonical

        # Load countries
        for country_data in data.get('countries', []):
            place = PlaceVariant(
                canonical=country_data['canonical'],
                country=country_data['canonical'],
                place_type='country',
                variants=country_data.get('variants', {})
            )
            self.place_variants.append(place)

            # Build lookup cache
            for variant in place.get_all_variants():
                self._place_lookup[variant] = place.canonical

    def normalize_suffix(self, suffix: str) -> Optional[str]:
        """
        Normalize a suffix to its canonical form.

        Args:
            suffix: The suffix to normalize (e.g., "duc", "herzog", "duke")

        Returns:
            Canonical form (e.g., "duke") or None if not found
        """
        if not suffix:
            return None

        suffix_lower = suffix.lower().strip()
        return self._title_lookup.get(suffix_lower)

    def normalize_place(self, place_name: str) -> Optional[str]:
        """
        Normalize a place name to its canonical form.

        Args:
            place_name: The place name to normalize (e.g., "Wien", "Vienna", "Bécs")

        Returns:
            Canonical form (e.g., "vienna") or None if not found
        """
        if not place_name:
            return None

        place_lower = place_name.lower().strip()
        return self._place_lookup.get(place_lower)

    def get_all_suffix_variants(self) -> Set[str]:
        """Get all suffix variants across all categories (normalized to lowercase)."""
        variants = set()
        for title in self.titles + self.professional_suffixes + self.generational_suffixes:
            variants.update(title.get_all_variants())
        return variants

    def get_noble_title_variants(self) -> Set[str]:
        """Get only noble title variants (excluding professional/generational)."""
        variants = set()
        for title in self.titles:
            variants.update(title.get_all_variants())
        return variants

    def get_suffix_rank(self, suffix: str) -> Optional[int]:
        """
        Get the rank of a noble title suffix.

        Args:
            suffix: The suffix (e.g., "duke", "count")

        Returns:
            Rank number (lower is higher rank) or None
        """
        canonical = self.normalize_suffix(suffix)
        if canonical:
            for title in self.titles:
                if title.canonical == canonical:
                    return title.rank
        return None

    def are_equivalent_suffixes(self, suffix1: str, suffix2: str) -> bool:
        """
        Check if two suffixes are equivalent (same canonical form).

        Args:
            suffix1: First suffix (e.g., "duke")
            suffix2: Second suffix (e.g., "duc")

        Returns:
            True if equivalent, False otherwise
        """
        norm1 = self.normalize_suffix(suffix1)
        norm2 = self.normalize_suffix(suffix2)
        return norm1 is not None and norm1 == norm2

    def are_equivalent_places(self, place1: str, place2: str) -> bool:
        """
        Check if two place names are equivalent (same canonical form).

        Args:
            place1: First place name (e.g., "Vienna")
            place2: Second place name (e.g., "Wien")

        Returns:
            True if equivalent, False otherwise
        """
        norm1 = self.normalize_place(place1)
        norm2 = self.normalize_place(place2)
        return norm1 is not None and norm1 == norm2

    def get_title_info(self, suffix: str) -> Optional[TitleVariant]:
        """
        Get full information about a title.

        Args:
            suffix: The suffix to look up

        Returns:
            TitleVariant object or None if not found
        """
        canonical = self.normalize_suffix(suffix)
        if canonical:
            for title in self.titles + self.professional_suffixes + self.generational_suffixes:
                if title.canonical == canonical:
                    return title
        return None

    def get_place_info(self, place_name: str) -> Optional[PlaceVariant]:
        """
        Get full information about a place.

        Args:
            place_name: The place name to look up

        Returns:
            PlaceVariant object or None if not found
        """
        canonical = self.normalize_place(place_name)
        if canonical:
            for place in self.place_variants:
                if place.canonical == canonical:
                    return place
        return None


# Create a singleton instance for easy import
reference_data = ReferenceDataLoader()
