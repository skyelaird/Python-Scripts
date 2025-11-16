"""Event class for representing GEDCOM events."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Event:
    """Represents a genealogical event (birth, death, marriage, etc.).

    Attributes:
        type: The type of event (e.g., 'BIRT', 'DEAT', 'MARR', 'BURI')
        date: The date of the event in GEDCOM format
        place: The place where the event occurred
        notes: Additional notes about the event
        sources: List of source citations for this event
        attributes: Additional GEDCOM attributes (e.g., AGE, CAUS)
    """

    type: str
    date: Optional[str] = None
    place: Optional[str] = None
    notes: Optional[str] = None
    sources: list = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a human-readable string representation of the event."""
        parts = [self.type]
        if self.date:
            parts.append(f"on {self.date}")
        if self.place:
            parts.append(f"at {self.place}")
        return " ".join(parts)

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return f"Event(type={self.type!r}, date={self.date!r}, place={self.place!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            'type': self.type,
            'date': self.date,
            'place': self.place,
            'notes': self.notes,
            'sources': self.sources,
            'attributes': self.attributes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create an Event instance from a dictionary.

        Args:
            data: Dictionary containing event data

        Returns:
            Event instance
        """
        return cls(
            type=data['type'],
            date=data.get('date'),
            place=data.get('place'),
            notes=data.get('notes'),
            sources=data.get('sources', []),
            attributes=data.get('attributes', {})
        )

    def is_valid(self) -> bool:
        """Check if the event has minimum required information.

        Returns:
            True if event has at least a type, False otherwise
        """
        return bool(self.type)

    def get_year(self) -> Optional[int]:
        """Extract year from date string if possible.

        Returns:
            Year as integer, or None if cannot be extracted
        """
        if not self.date:
            return None

        # Try to find a 4-digit year in the date string
        import re
        year_match = re.search(r'\b(\d{4})\b', self.date)
        if year_match:
            return int(year_match.group(1))
        return None
