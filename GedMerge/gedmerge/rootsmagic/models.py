"""Data models for RootsMagic database entities."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class RMName:
    """Represents a name record from the NameTable."""
    name_id: int
    owner_id: int
    surname: Optional[str] = None
    given: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    nickname: Optional[str] = None
    name_type: int = 0
    date: Optional[str] = None
    sort_date: Optional[int] = None
    is_primary: bool = False
    is_private: bool = False
    proof: int = 0
    sentence: Optional[str] = None
    note: Optional[str] = None
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    display: int = 0
    language: Optional[str] = None
    utc_mod_date: Optional[float] = None
    surname_mp: Optional[str] = None  # Metaphone encoding
    given_mp: Optional[str] = None    # Metaphone encoding
    nickname_mp: Optional[str] = None # Metaphone encoding

    def full_name(self) -> str:
        """Return formatted full name."""
        parts = []
        if self.prefix:
            parts.append(self.prefix)
        if self.given:
            parts.append(self.given)
        if self.surname:
            parts.append(self.surname)
        if self.suffix:
            parts.append(self.suffix)
        return ' '.join(parts) if parts else 'Unknown'

    def gedcom_name(self) -> str:
        """Return GEDCOM-style name: Given /Surname/"""
        given = self.given or ''
        surname = f'/{self.surname}/' if self.surname else ''
        return f'{given} {surname}'.strip()


@dataclass
class RMEvent:
    """Represents an event record from the EventTable."""
    event_id: int
    event_type: int
    owner_type: int
    owner_id: int
    family_id: Optional[int] = None
    place_id: Optional[int] = None
    site_id: Optional[int] = None
    date: Optional[str] = None
    sort_date: Optional[int] = None
    is_primary: bool = False
    is_private: bool = False
    proof: int = 0
    status: int = 0
    sentence: Optional[str] = None
    details: Optional[str] = None
    note: Optional[str] = None
    utc_mod_date: Optional[float] = None

    # Common event types
    BIRTH = 1
    DEATH = 2
    BURIAL = 3
    MARRIAGE = 4
    DIVORCE = 5

    def get_year(self) -> Optional[int]:
        """Extract year from date string."""
        if not self.date:
            return None
        import re
        year_match = re.search(r'\b(\d{4})\b', self.date)
        return int(year_match.group(1)) if year_match else None


@dataclass
class RMPlace:
    """Represents a place record from the PlaceTable."""
    place_id: int
    place_type: int = 0
    name: Optional[str] = None
    abbrev: Optional[str] = None
    normalized: Optional[str] = None
    latitude: Optional[int] = None
    longitude: Optional[int] = None
    lat_long_exact: bool = False
    master_id: Optional[int] = None
    note: Optional[str] = None
    reverse: Optional[str] = None
    fs_id: Optional[int] = None
    an_id: Optional[int] = None
    utc_mod_date: Optional[float] = None


@dataclass
class RMSource:
    """Represents a source record from the SourceTable."""
    source_id: int
    name: Optional[str] = None
    ref_number: Optional[str] = None
    actual_text: Optional[str] = None
    comments: Optional[str] = None
    is_private: bool = False
    template_id: Optional[int] = None
    fields: Optional[bytes] = None
    utc_mod_date: Optional[float] = None


@dataclass
class RMCitation:
    """Represents a citation record from the CitationTable."""
    citation_id: int
    source_id: int
    comments: Optional[str] = None
    actual_text: Optional[str] = None
    ref_number: Optional[str] = None
    footnote: Optional[str] = None
    short_footnote: Optional[str] = None
    bibliography: Optional[str] = None
    fields: Optional[bytes] = None
    utc_mod_date: Optional[float] = None
    citation_name: Optional[str] = None


@dataclass
class RMFamily:
    """Represents a family record from the FamilyTable."""
    family_id: int
    father_id: Optional[int] = None
    mother_id: Optional[int] = None
    child_id: Optional[int] = None
    husb_order: int = 0
    wife_order: int = 0
    is_private: bool = False
    proof: int = 0
    spouse_label: int = 0
    father_label: int = 0
    mother_label: int = 0
    spouse_label_str: Optional[str] = None
    father_label_str: Optional[str] = None
    mother_label_str: Optional[str] = None
    note: Optional[str] = None
    utc_mod_date: Optional[float] = None


@dataclass
class RMPerson:
    """Represents a person record from the PersonTable with associated data."""
    person_id: int
    unique_id: Optional[str] = None
    sex: int = 0  # 0=Unknown, 1=Male, 2=Female
    parent_id: Optional[int] = None
    spouse_id: Optional[int] = None
    color: int = 0
    color1: int = 0
    color2: int = 0
    color3: int = 0
    color4: int = 0
    color5: int = 0
    color6: int = 0
    color7: int = 0
    color8: int = 0
    color9: int = 0
    relate1: int = 0
    relate2: int = 0
    flags: int = 0
    living: bool = False
    is_private: bool = False
    proof: int = 0
    bookmark: bool = False
    note: Optional[str] = None
    utc_mod_date: Optional[float] = None

    # Associated records (loaded separately)
    names: List[RMName] = field(default_factory=list)
    events: List[RMEvent] = field(default_factory=list)

    def get_sex_string(self) -> str:
        """Convert sex code to string."""
        return {0: 'U', 1: 'M', 2: 'F'}.get(self.sex, 'U')

    def get_primary_name(self) -> Optional[RMName]:
        """Get the primary name record."""
        for name in self.names:
            if name.is_primary:
                return name
        return self.names[0] if self.names else None

    def get_birth_event(self) -> Optional[RMEvent]:
        """Get the birth event."""
        for event in self.events:
            if event.event_type == RMEvent.BIRTH and event.is_primary:
                return event
        return None

    def get_death_event(self) -> Optional[RMEvent]:
        """Get the death event."""
        for event in self.events:
            if event.event_type == RMEvent.DEATH and event.is_primary:
                return event
        return None

    def get_birth_year(self) -> Optional[int]:
        """Get birth year."""
        birth = self.get_birth_event()
        return birth.get_year() if birth else None

    def get_death_year(self) -> Optional[int]:
        """Get death year."""
        death = self.get_death_event()
        return death.get_year() if death else None

    def __str__(self) -> str:
        """Return human-readable representation."""
        name = self.get_primary_name()
        name_str = name.full_name() if name else 'Unknown'
        birth_year = self.get_birth_year()
        death_year = self.get_death_year()

        if birth_year and death_year:
            return f"{name_str} ({birth_year}-{death_year})"
        elif birth_year:
            return f"{name_str} (b. {birth_year})"
        elif death_year:
            return f"{name_str} (d. {death_year})"
        else:
            return name_str
