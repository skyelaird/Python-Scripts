"""
Match scoring engine with confidence-based decisions.

Calculates similarity scores across multiple dimensions and combines
them into an overall confidence score.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from rapidfuzz import fuzz
import phonetics
from ..rootsmagic.models import RMPerson, RMName, RMEvent


@dataclass
class MatchResult:
    """Results of matching two person records."""

    # Individual component scores (0-100)
    name_score: float = 0.0
    phonetic_score: float = 0.0
    date_score: float = 0.0
    place_score: float = 0.0
    relationship_score: float = 0.0
    sex_score: float = 0.0

    # Overall confidence (0-100)
    overall_score: float = 0.0

    # Detailed breakdown
    details: Dict[str, any] = field(default_factory=dict)

    # Flags
    is_exact_name_match: bool = False
    is_exact_date_match: bool = False
    has_conflicting_info: bool = False

    def __str__(self) -> str:
        """Human-readable description."""
        return (
            f"Match Score: {self.overall_score:.1f}%\n"
            f"  Name: {self.name_score:.1f}%\n"
            f"  Phonetic: {self.phonetic_score:.1f}%\n"
            f"  Dates: {self.date_score:.1f}%\n"
            f"  Places: {self.place_score:.1f}%\n"
            f"  Relationships: {self.relationship_score:.1f}%\n"
            f"  Sex: {self.sex_score:.1f}%"
        )


class MatchScorer:
    """
    Calculates match scores between person records.

    Scoring weights:
    - Name similarity: 35%
    - Phonetic matching: 25%
    - Date proximity: 20%
    - Place matching: 10%
    - Relationship overlap: 8%
    - Sex match: 2%
    """

    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        'name': 0.35,
        'phonetic': 0.25,
        'date': 0.20,
        'place': 0.10,
        'relationship': 0.08,
        'sex': 0.02,
    }

    # Date tolerance (years)
    DATE_EXACT_TOLERANCE = 0  # Same year = exact
    DATE_CLOSE_TOLERANCE = 2  # Within 2 years = close
    DATE_LIKELY_TOLERANCE = 5  # Within 5 years = possible

    # Multilingual language equivalents
    LANGUAGE_GROUPS = {
        'germanic': {'en', 'de', 'nl', 'sv', 'no', 'da'},
        'romance': {'fr', 'es', 'it', 'pt', 'ro', 'la'},
        'slavic': {'ru', 'pl', 'cs', 'uk', 'sr', 'hr'},
    }

    def calculate_match_score(
        self,
        person1: RMPerson,
        person2: RMPerson
    ) -> MatchResult:
        """
        Calculate overall match score between two persons.

        Args:
            person1: First person
            person2: Second person

        Returns:
            MatchResult with scores and details
        """
        result = MatchResult()

        # Calculate individual component scores
        result.name_score = self._score_names(person1, person2, result)
        result.phonetic_score = self._score_phonetic(person1, person2, result)
        result.date_score = self._score_dates(person1, person2, result)
        result.place_score = self._score_places(person1, person2, result)
        result.relationship_score = self._score_relationships(person1, person2, result)
        result.sex_score = self._score_sex(person1, person2, result)

        # Calculate weighted overall score
        result.overall_score = (
            result.name_score * self.WEIGHTS['name'] +
            result.phonetic_score * self.WEIGHTS['phonetic'] +
            result.date_score * self.WEIGHTS['date'] +
            result.place_score * self.WEIGHTS['place'] +
            result.relationship_score * self.WEIGHTS['relationship'] +
            result.sex_score * self.WEIGHTS['sex']
        )

        # Apply penalties for conflicting information
        if result.has_conflicting_info:
            result.overall_score *= 0.5  # 50% penalty
            result.details['penalty'] = 'Conflicting information detected'

        return result

    def _score_names(
        self,
        person1: RMPerson,
        person2: RMPerson,
        result: MatchResult
    ) -> float:
        """
        Score name similarity using fuzzy matching.

        Handles:
        - Multiple name variations per person
        - Multilingual names (same person, different languages)
        - Given name vs nickname matching
        - Surname variations
        """
        if not person1.names or not person2.names:
            return 0.0

        max_score = 0.0
        best_match = None

        # Compare all name combinations
        for name1 in person1.names:
            for name2 in person2.names:
                score = self._compare_names(name1, name2)
                if score > max_score:
                    max_score = score
                    best_match = (name1, name2)

        # Store details
        if best_match:
            result.details['name_match'] = {
                'name1': f"{best_match[0].given} /{best_match[0].surname}/",
                'name2': f"{best_match[1].given} /{best_match[1].surname}/",
                'score': max_score,
            }

            if max_score == 100:
                result.is_exact_name_match = True

        return max_score

    def _compare_names(self, name1: RMName, name2: RMName) -> float:
        """
        Compare two name records.

        Returns score 0-100.
        """
        scores = []

        # Compare surnames (most important)
        if name1.surname and name2.surname:
            surname_score = fuzz.ratio(
                name1.surname.lower(),
                name2.surname.lower()
            )
            scores.append(surname_score * 1.2)  # Weight surnames more

        # Compare given names
        if name1.given and name2.given:
            given_score = fuzz.ratio(
                name1.given.lower(),
                name2.given.lower()
            )
            scores.append(given_score)

        # Compare nicknames if present
        if name1.nickname and name2.nickname:
            nickname_score = fuzz.ratio(
                name1.nickname.lower(),
                name2.nickname.lower()
            )
            scores.append(nickname_score)

        # Cross-check: nickname vs given name
        if name1.nickname and name2.given:
            cross_score = fuzz.ratio(
                name1.nickname.lower(),
                name2.given.lower()
            )
            scores.append(cross_score * 0.9)  # Slightly lower weight

        if name1.given and name2.nickname:
            cross_score = fuzz.ratio(
                name1.given.lower(),
                name2.nickname.lower()
            )
            scores.append(cross_score * 0.9)

        if not scores:
            return 0.0

        # Return average of all comparisons, capped at 100
        avg_score = sum(scores) / len(scores)
        return min(avg_score, 100.0)

    def _score_phonetic(
        self,
        person1: RMPerson,
        person2: RMPerson,
        result: MatchResult
    ) -> float:
        """
        Score phonetic similarity using Metaphone.

        Handles spelling variations and pronunciation-based matching.
        """
        if not person1.names or not person2.names:
            return 0.0

        max_score = 0.0

        for name1 in person1.names:
            for name2 in person2.names:
                score = self._compare_phonetic(name1, name2)
                max_score = max(max_score, score)

        return max_score

    def _compare_phonetic(self, name1: RMName, name2: RMName) -> float:
        """Compare phonetic encodings of names."""
        matches = 0
        total = 0

        # Compare surname phonetics
        if name1.surname_mp and name2.surname_mp:
            total += 1
            if name1.surname_mp == name2.surname_mp:
                matches += 2  # Surnames count double

        # Compare given name phonetics
        if name1.given_mp and name2.given_mp:
            total += 1
            if name1.given_mp == name2.given_mp:
                matches += 1

        # Compare nickname phonetics
        if name1.nickname_mp and name2.nickname_mp:
            total += 1
            if name1.nickname_mp == name2.nickname_mp:
                matches += 1

        if total == 0:
            return 0.0

        # Normalize to 0-100
        return (matches / (total + 1)) * 100  # +1 to account for double weight

    def _score_dates(
        self,
        person1: RMPerson,
        person2: RMPerson,
        result: MatchResult
    ) -> float:
        """
        Score date proximity for birth and death dates.

        Scoring:
        - Same year: 100%
        - Within 2 years: 80%
        - Within 5 years: 50%
        - Beyond 5 years: 0%
        """
        scores = []

        # Compare birth dates
        birth1 = self._get_event_year(person1, 'Birth')
        birth2 = self._get_event_year(person2, 'Birth')

        if birth1 and birth2:
            birth_score = self._compare_years(birth1, birth2)
            scores.append(birth_score)

            if birth1 == birth2:
                result.details['birth_match'] = 'exact'
            elif abs(birth1 - birth2) <= self.DATE_CLOSE_TOLERANCE:
                result.details['birth_match'] = 'close'

        # Compare death dates
        death1 = self._get_event_year(person1, 'Death')
        death2 = self._get_event_year(person2, 'Death')

        if death1 and death2:
            death_score = self._compare_years(death1, death2)
            scores.append(death_score)

            if death1 == death2:
                result.details['death_match'] = 'exact'
            elif abs(death1 - death2) <= self.DATE_CLOSE_TOLERANCE:
                result.details['death_match'] = 'close'

        # Check for conflicts (dates too far apart)
        if birth1 and birth2 and abs(birth1 - birth2) > 10:
            result.has_conflicting_info = True
        if death1 and death2 and abs(death1 - death2) > 10:
            result.has_conflicting_info = True

        if not scores:
            return 50.0  # Neutral score if no dates available

        # If both dates match exactly
        if len(scores) == 2 and all(s == 100 for s in scores):
            result.is_exact_date_match = True

        return sum(scores) / len(scores)

    def _compare_years(self, year1: int, year2: int) -> float:
        """Compare two years and return similarity score."""
        diff = abs(year1 - year2)

        if diff == self.DATE_EXACT_TOLERANCE:
            return 100.0
        elif diff <= self.DATE_CLOSE_TOLERANCE:
            return 80.0
        elif diff <= self.DATE_LIKELY_TOLERANCE:
            return 50.0
        else:
            return 0.0

    def _score_places(
        self,
        person1: RMPerson,
        person2: RMPerson,
        result: MatchResult
    ) -> float:
        """
        Score place similarity for birth and death places.

        Handles multilingual place names.
        """
        scores = []

        # Compare birth places
        birth_place1 = self._get_event_place(person1, 'Birth')
        birth_place2 = self._get_event_place(person2, 'Birth')

        if birth_place1 and birth_place2:
            place_score = self._compare_places(birth_place1, birth_place2)
            scores.append(place_score)

        # Compare death places
        death_place1 = self._get_event_place(person1, 'Death')
        death_place2 = self._get_event_place(person2, 'Death')

        if death_place1 and death_place2:
            place_score = self._compare_places(death_place1, death_place2)
            scores.append(place_score)

        if not scores:
            return 50.0  # Neutral score

        return sum(scores) / len(scores)

    def _compare_places(self, place1: str, place2: str) -> float:
        """Compare two place names using fuzzy matching."""
        if not place1 or not place2:
            return 0.0

        # Direct fuzzy match
        score = fuzz.token_sort_ratio(place1.lower(), place2.lower())

        return float(score)

    def _score_relationships(
        self,
        person1: RMPerson,
        person2: RMPerson,
        result: MatchResult
    ) -> float:
        """
        Score relationship overlap.

        If two persons share the same parents or spouses, they're more
        likely to be duplicates.
        """
        # Check for shared spouses
        spouses1 = set(person1.spouse_family_ids or [])
        spouses2 = set(person2.spouse_family_ids or [])
        shared_spouses = spouses1 & spouses2

        # Check for shared parents
        parents1 = set(person1.parent_family_ids or [])
        parents2 = set(person2.parent_family_ids or [])
        shared_parents = parents1 & parents2

        # Calculate score
        score = 0.0

        if shared_parents:
            score += 60.0  # Same parents = strong indicator
            result.details['shared_parents'] = len(shared_parents)

        if shared_spouses:
            score += 40.0  # Same spouse = strong indicator
            result.details['shared_spouses'] = len(shared_spouses)

        return min(score, 100.0)

    def _score_sex(
        self,
        person1: RMPerson,
        person2: RMPerson,
        result: MatchResult
    ) -> float:
        """
        Score sex/gender match.

        Same sex: 100%
        Unknown sex: 50% (neutral)
        Different sex: 0% (strong conflict)
        """
        sex1 = person1.sex
        sex2 = person2.sex

        # If either is unknown
        if not sex1 or sex1 == 'U' or not sex2 or sex2 == 'U':
            return 50.0

        # If same
        if sex1 == sex2:
            return 100.0

        # If different (strong conflict)
        result.has_conflicting_info = True
        result.details['sex_conflict'] = f"{sex1} vs {sex2}"
        return 0.0

    def _get_event_year(self, person: RMPerson, event_type: str) -> Optional[int]:
        """Extract year from event."""
        if not person.events:
            return None

        for event in person.events:
            if event.event_type == event_type and event.date:
                # Try to extract year from date string
                try:
                    # Handle various date formats
                    # Simple year extraction: look for 4-digit number
                    import re
                    match = re.search(r'\b(\d{4})\b', event.date)
                    if match:
                        return int(match.group(1))
                except:
                    pass

        return None

    def _get_event_place(self, person: RMPerson, event_type: str) -> Optional[str]:
        """Extract place from event."""
        if not person.events:
            return None

        for event in person.events:
            if event.event_type == event_type and event.place:
                return event.place

        return None
