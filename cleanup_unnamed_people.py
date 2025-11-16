#!/usr/bin/env python3
"""
Clean up unnamed and placeholder people from RootsMagic genealogy database.

This script identifies and removes:
1. Unnamed people without at least one named ancestor
2. EndofLine people with just the surname of the child (e.g., "Smith" being father of "John Smith")
3. "MRS" as a mother/spouse unless otherwise connected to ancestors
4. MRS clones of husband's name (e.g., "Mrs. John Smith")

Note: MRS persons who are named elsewhere are flagged as potential merge candidates.
"""

import sys
import sqlite3
import argparse
import re
import unicodedata
from pathlib import Path
from typing import Set, Dict, List, Optional, Tuple
from collections import defaultdict

# Add GedMerge to path
sys.path.insert(0, str(Path(__file__).parent / "GedMerge"))

from gedmerge.rootsmagic.adapter import RootsMagicDatabase
from gedmerge.rootsmagic.models import RMPerson, RMName, RMFamily


# Multi-language placeholder titles for Western European languages
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

# Prefixes that indicate placeholder names (e.g., "Mrs. John Smith")
PLACEHOLDER_PREFIXES = {
    'MRS.', 'MRS ', 'MS.', 'MS ', 'MISS ', 'MR.', 'MR ',
    'MME.', 'MME ', 'M.', 'M ', 'MLLE.', 'MLLE ',
    'SRA.', 'SRA ', 'SR.', 'SR ', 'SRTA.', 'SRTA ',
    'FRAU ', 'HERR ',
    'MEVROUW ', 'MEVR.', 'MEVR ', 'MIJNHEER ', 'DHR.', 'DHR ',
    'SIG.RA ', 'SIG.', 'SIG ',
}


def normalize_for_comparison(text: str) -> str:
    """
    Normalize text for language-agnostic comparison.

    This removes accents and converts to uppercase for comparison,
    so "Señora" matches "SENORA", "Madame" matches "MADAME", etc.

    Args:
        text: Text to normalize

    Returns:
        Normalized uppercase text without accents
    """
    if not text:
        return ''

    # Normalize unicode to decomposed form (NFD)
    # This separates accented characters into base + accent
    nfd = unicodedata.normalize('NFD', text)

    # Remove accent marks (combining characters)
    without_accents = ''.join(
        char for char in nfd
        if unicodedata.category(char) != 'Mn'  # Mn = Mark, Nonspacing
    )

    return without_accents.upper().strip()


class PersonCleaner:
    """Identifies and removes unnamed/placeholder people from genealogy database."""

    def __init__(self, db: RootsMagicDatabase):
        self.db = db
        self.all_persons: Dict[int, RMPerson] = {}
        self.families: List[RMFamily] = []
        self.parent_child_map: Dict[int, List[int]] = defaultdict(list)  # parent_id -> [child_ids]
        self.child_parent_map: Dict[int, List[int]] = defaultdict(list)  # child_id -> [parent_ids]
        self.to_delete: Set[int] = set()
        self.merge_candidates: Set[int] = set()
        self.deletion_reasons: Dict[int, str] = {}

    def load_data(self):
        """Load all persons and families from the database."""
        print("Loading database...")

        # Load all persons
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT PersonID FROM PersonTable ORDER BY PersonID")
        person_ids = [row[0] for row in cursor.fetchall()]

        for person_id in person_ids:
            person = self.db.get_person(person_id, load_names=True, load_events=False)
            if person:
                self.all_persons[person_id] = person

        # Load all families
        cursor.execute("SELECT FamilyID FROM FamilyTable ORDER BY FamilyID")
        family_ids = [row[0] for row in cursor.fetchall()]

        for family_id in family_ids:
            family = self.db.get_family(family_id)
            if family:
                self.families.append(family)

        # Build parent-child relationships
        cursor.execute("""
            SELECT FamilyID, ChildID FROM ChildTable ORDER BY FamilyID, ChildOrder
        """)

        family_children = defaultdict(list)
        for family_id, child_id in cursor.fetchall():
            family_children[family_id].append(child_id)

        # Map parents to children
        for family in self.families:
            children = family_children.get(family.family_id, [])

            for child_id in children:
                if family.father_id:
                    self.parent_child_map[family.father_id].append(child_id)
                    self.child_parent_map[child_id].append(family.father_id)

                if family.mother_id:
                    self.parent_child_map[family.mother_id].append(child_id)
                    self.child_parent_map[child_id].append(family.mother_id)

        print(f"Loaded {len(self.all_persons)} persons and {len(self.families)} families")

    def is_unnamed(self, person: RMPerson) -> bool:
        """Check if a person is unnamed (no given name)."""
        primary_name = person.get_primary_name()
        if not primary_name:
            return True

        # No given name or given name is empty/whitespace
        if not primary_name.given or not primary_name.given.strip():
            return True

        return False

    def is_mrs_placeholder(self, person: RMPerson) -> Tuple[bool, Optional[str]]:
        """
        Check if a person is a placeholder title (MRS, Madame, Señora, etc.).

        Supports multi-language detection for Western European languages.

        Returns:
            Tuple of (is_placeholder, reason)
        """
        primary_name = person.get_primary_name()
        if not primary_name:
            return False, None

        given = (primary_name.given or '').strip()
        surname = (primary_name.surname or '').strip()

        # Normalize for comparison (removes accents, uppercase)
        given_normalized = normalize_for_comparison(given)

        # Check for placeholder title as given name (multi-language)
        if given_normalized in PLACEHOLDER_TITLES:
            return True, f"Generic placeholder title '{given}' with surname '{surname}'"

        # Check for "Mrs./Mme./Sra. [Full Name]" pattern (prefix + more)
        for prefix in PLACEHOLDER_PREFIXES:
            if given_normalized.startswith(prefix):
                # If there's more after the prefix, it's likely a clone name
                remainder = given_normalized[len(prefix):].strip()
                if remainder:
                    return True, f"Placeholder clone name: '{primary_name.full_name()}'"

        return False, None

    def is_endofline_parent(self, person: RMPerson) -> Tuple[bool, Optional[str]]:
        """
        Check if person is an EndofLine parent (surname only matching child's surname).

        Uses Unicode normalization for language-agnostic surname matching.

        Returns:
            Tuple of (is_endofline, reason)
        """
        primary_name = person.get_primary_name()
        if not primary_name:
            return False, None

        given = (primary_name.given or '').strip()
        surname = (primary_name.surname or '').strip()

        # Must have surname but no real given name
        if not surname or given:
            return False, None

        # Normalize surname for comparison
        surname_normalized = normalize_for_comparison(surname)

        # Check if this person's surname matches their child's surname
        children_ids = self.parent_child_map.get(person.person_id, [])
        if not children_ids:
            return False, None

        for child_id in children_ids:
            child = self.all_persons.get(child_id)
            if not child:
                continue

            child_name = child.get_primary_name()
            if not child_name:
                continue

            child_surname = (child_name.surname or '').strip()
            child_given = (child_name.given or '').strip()

            # Normalize child surname for comparison
            child_surname_normalized = normalize_for_comparison(child_surname)

            # If parent surname matches child surname and child has a given name
            if surname_normalized == child_surname_normalized and child_given:
                return True, f"EndofLine parent '{surname}' of child '{child_name.full_name()}'"

        return False, None

    def has_named_ancestor(self, person_id: int, visited: Optional[Set[int]] = None) -> bool:
        """
        Recursively check if person has at least one named ancestor.

        Args:
            person_id: ID of person to check
            visited: Set of already visited person IDs (to prevent infinite loops)

        Returns:
            True if person has at least one named ancestor
        """
        if visited is None:
            visited = set()

        if person_id in visited:
            return False

        visited.add(person_id)

        # Get parents
        parent_ids = self.child_parent_map.get(person_id, [])

        if not parent_ids:
            # No parents, so no ancestors
            return False

        for parent_id in parent_ids:
            parent = self.all_persons.get(parent_id)
            if not parent:
                continue

            # Check if this parent is named
            if not self.is_unnamed(parent):
                primary_name = parent.get_primary_name()
                given = (primary_name.given or '').strip() if primary_name else ''

                # Make sure it's not just a placeholder name
                is_mrs, _ = self.is_mrs_placeholder(parent)
                is_eol, _ = self.is_endofline_parent(parent)

                if given and not is_mrs and not is_eol:
                    return True

            # Recursively check parent's ancestors
            if self.has_named_ancestor(parent_id, visited):
                return True

        return False

    def check_mrs_named_elsewhere(self, person: RMPerson) -> bool:
        """
        Check if a placeholder person is named elsewhere (potential merge candidate).

        This checks if there's another person with the same surname but a real given name.
        Uses Unicode normalization for language-agnostic matching.
        """
        primary_name = person.get_primary_name()
        if not primary_name or not primary_name.surname:
            return False

        surname = primary_name.surname.strip()
        surname_normalized = normalize_for_comparison(surname)

        # Search for other people with same surname and a real given name
        for other_id, other_person in self.all_persons.items():
            if other_id == person.person_id:
                continue

            other_name = other_person.get_primary_name()
            if not other_name:
                continue

            other_surname = (other_name.surname or '').strip()
            other_given = (other_name.given or '').strip()

            # Normalize for comparison
            other_surname_normalized = normalize_for_comparison(other_surname)
            other_given_normalized = normalize_for_comparison(other_given)

            # Same surname, has a real given name (not a placeholder), and same sex
            if (surname_normalized == other_surname_normalized and
                other_given and
                other_given_normalized not in PLACEHOLDER_TITLES and
                person.sex == other_person.sex):
                return True

        return False

    def identify_deletions(self):
        """Identify all persons to be deleted based on criteria."""
        print("\nAnalyzing persons...")

        unnamed_count = 0
        mrs_count = 0
        eol_count = 0

        for person_id, person in self.all_persons.items():
            primary_name = person.get_primary_name()
            name_str = primary_name.full_name() if primary_name else "Unknown"

            # Check if unnamed
            is_unnamed = self.is_unnamed(person)

            # Check if MRS placeholder
            is_mrs, mrs_reason = self.is_mrs_placeholder(person)

            # Check if EndofLine parent
            is_eol, eol_reason = self.is_endofline_parent(person)

            # Apply deletion rules
            should_delete = False
            reason = None

            if is_eol:
                should_delete = True
                reason = eol_reason
                eol_count += 1
            elif is_mrs:
                # Check if MRS is named elsewhere (merge candidate)
                if self.check_mrs_named_elsewhere(person):
                    self.merge_candidates.add(person_id)
                    print(f"  Merge candidate: {name_str} (ID: {person_id}) - {mrs_reason}")
                else:
                    # Only delete MRS if not connected to named ancestors
                    if not self.has_named_ancestor(person_id):
                        should_delete = True
                        reason = f"{mrs_reason} - no named ancestors"
                        mrs_count += 1
            elif is_unnamed:
                # Only delete unnamed if they don't have named ancestors
                if not self.has_named_ancestor(person_id):
                    should_delete = True
                    reason = f"Unnamed person '{name_str}' - no named ancestors"
                    unnamed_count += 1

            if should_delete and reason:
                self.to_delete.add(person_id)
                self.deletion_reasons[person_id] = reason

        print(f"\nFound {len(self.to_delete)} persons to delete:")
        print(f"  - {unnamed_count} unnamed without named ancestors")
        print(f"  - {mrs_count} MRS placeholders without named ancestors")
        print(f"  - {eol_count} EndofLine surname-only parents")
        print(f"\nFound {len(self.merge_candidates)} potential merge candidates")

    def print_deletion_report(self, limit: int = 50):
        """Print a detailed report of persons to be deleted."""
        print("\n" + "="*80)
        print("DELETION REPORT")
        print("="*80)

        if not self.to_delete:
            print("No persons to delete.")
            return

        print(f"\nShowing first {min(limit, len(self.to_delete))} of {len(self.to_delete)} persons to delete:\n")

        for i, person_id in enumerate(sorted(self.to_delete)[:limit]):
            person = self.all_persons.get(person_id)
            if not person:
                continue

            primary_name = person.get_primary_name()
            name_str = primary_name.full_name() if primary_name else "Unknown"
            reason = self.deletion_reasons.get(person_id, "Unknown reason")

            # Get children
            children = self.parent_child_map.get(person_id, [])
            children_names = []
            for child_id in children[:3]:  # Show first 3 children
                child = self.all_persons.get(child_id)
                if child and child.get_primary_name():
                    children_names.append(child.get_primary_name().full_name())

            children_str = ""
            if children_names:
                children_str = f", parent of: {', '.join(children_names)}"
                if len(children) > 3:
                    children_str += f" and {len(children)-3} more"

            print(f"{i+1}. ID {person_id}: {name_str} ({person.get_sex_string()})")
            print(f"   Reason: {reason}{children_str}")

        if len(self.to_delete) > limit:
            print(f"\n... and {len(self.to_delete) - limit} more")

        # Print merge candidates
        if self.merge_candidates:
            print("\n" + "="*80)
            print("POTENTIAL MERGE CANDIDATES (MRS persons with possible matches)")
            print("="*80)
            print(f"\nShowing first {min(limit, len(self.merge_candidates))} candidates:\n")

            for i, person_id in enumerate(sorted(self.merge_candidates)[:limit]):
                person = self.all_persons.get(person_id)
                if not person:
                    continue

                primary_name = person.get_primary_name()
                name_str = primary_name.full_name() if primary_name else "Unknown"

                print(f"{i+1}. ID {person_id}: {name_str} - Review for potential merge")

    def delete_persons(self, dry_run: bool = True):
        """
        Delete identified persons from the database.

        Args:
            dry_run: If True, only simulate deletion without making changes
        """
        if not self.to_delete:
            print("No persons to delete.")
            return

        if dry_run:
            print("\n[DRY RUN] Would delete the following persons:")
            self.print_deletion_report(limit=100)
            return

        print(f"\nDeleting {len(self.to_delete)} persons...")

        with self.db.transaction():
            cursor = self.db.conn.cursor()

            for person_id in self.to_delete:
                # Delete from NameTable
                cursor.execute("DELETE FROM NameTable WHERE OwnerID = ?", (person_id,))

                # Delete from EventTable
                cursor.execute("DELETE FROM EventTable WHERE OwnerType = 0 AND OwnerID = ?", (person_id,))

                # Remove from families (set to NULL)
                cursor.execute("UPDATE FamilyTable SET FatherID = NULL WHERE FatherID = ?", (person_id,))
                cursor.execute("UPDATE FamilyTable SET MotherID = NULL WHERE MotherID = ?", (person_id,))

                # Remove from ChildTable
                cursor.execute("DELETE FROM ChildTable WHERE ChildID = ?", (person_id,))

                # Delete from PersonTable
                cursor.execute("DELETE FROM PersonTable WHERE PersonID = ?", (person_id,))

        print(f"Successfully deleted {len(self.to_delete)} persons.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up unnamed and placeholder people from RootsMagic database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes without making them)
  python cleanup_unnamed_people.py database.rmtree --dry-run

  # Actually delete the persons
  python cleanup_unnamed_people.py database.rmtree --execute

  # Show detailed report with more entries
  python cleanup_unnamed_people.py database.rmtree --dry-run --report-limit 100
        """
    )

    parser.add_argument(
        'database',
        type=str,
        help='Path to RootsMagic .rmtree database file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Preview changes without making them (default)'
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete the persons (use with caution!)'
    )

    parser.add_argument(
        '--report-limit',
        type=int,
        default=50,
        help='Maximum number of entries to show in report (default: 50)'
    )

    args = parser.parse_args()

    # Validate database file
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)

    # If execute is specified, disable dry-run
    dry_run = not args.execute

    if not dry_run:
        response = input("\nWARNING: This will permanently delete persons from the database.\n"
                        "Are you sure you want to proceed? Type 'yes' to continue: ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)

    try:
        # Open database
        with RootsMagicDatabase(db_path) as db:
            cleaner = PersonCleaner(db)

            # Load data
            cleaner.load_data()

            # Identify deletions
            cleaner.identify_deletions()

            # Print report
            cleaner.print_deletion_report(limit=args.report_limit)

            # Delete (or simulate)
            cleaner.delete_persons(dry_run=dry_run)

            if not dry_run:
                print("\nDeletion complete. Database has been modified.")
                print("It is recommended to verify the changes in RootsMagic.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
