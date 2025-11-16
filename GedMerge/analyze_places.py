#!/usr/bin/env python3
"""Analyze place data quality issues in RootsMagic database."""

from gedmerge.rootsmagic import RootsMagicDatabase
from pathlib import Path
import zipfile
import re
from collections import defaultdict


def extract_db():
    """Extract database if needed."""
    zip_path = Path('/home/user/Python-Scripts/GedMerge/Rootsmagic/Joel2020.zip')
    extract_dir = zip_path.parent
    db_path = extract_dir / 'Joel2020.rmtree'

    if not db_path.exists():
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    return db_path


def analyze_places():
    """Analyze place data quality issues."""
    db_path = extract_db()

    with RootsMagicDatabase(db_path) as db:
        cursor = db.conn.cursor()

        # Get all places
        cursor.execute("SELECT PlaceID, Name, Normalized FROM PlaceTable ORDER BY PlaceID")
        places = cursor.fetchall()

        print(f"=== ANALYZING {len(places)} PLACES ===\n")

        # Issues to detect
        issues = {
            'parentheses': [],
            'semicolons': [],
            'lowercase_of': [],
            'mixed_case': [],
            'duplicates': defaultdict(list),
            'very_long': [],
            'contains_ruler': [],
            'contains_see': [],
        }

        for place in places:
            place_id, name, normalized = place

            if not name:
                continue

            # Check for parentheses (often comments)
            if '(' in name or ')' in name:
                issues['parentheses'].append((place_id, name))

            # Check for semicolons (often comments/notes)
            if ';' in name:
                issues['semicolons'].append((place_id, name))

            # Check for " of " (pet peeve)
            if ' of ' in name:
                issues['lowercase_of'].append((place_id, name))

            # Check for mixed case in same field
            words = name.split(',')
            for word in words:
                word = word.strip()
                if word and word[0].islower() and any(c.isupper() for c in word[1:]):
                    issues['mixed_case'].append((place_id, name))
                    break

            # Check for very long names (likely have comments)
            if len(name) > 100:
                issues['very_long'].append((place_id, name))

            # Check for genealogy-specific crud
            if 'ruler' in name.lower():
                issues['contains_ruler'].append((place_id, name))

            if 'see ' in name.lower():
                issues['contains_see'].append((place_id, name))

            # Track duplicates (case-insensitive)
            normalized_key = name.lower().strip()
            issues['duplicates'][normalized_key].append((place_id, name))

        # Filter duplicates to only actual duplicates
        issues['duplicates'] = {
            k: v for k, v in issues['duplicates'].items()
            if len(v) > 1
        }

        # Report findings
        print("=== DATA QUALITY ISSUES ===\n")

        print(f"Places with parentheses (likely comments): {len(issues['parentheses'])}")
        for place_id, name in issues['parentheses'][:10]:
            print(f"  {place_id}: {name}")
        if len(issues['parentheses']) > 10:
            print(f"  ... and {len(issues['parentheses']) - 10} more")

        print(f"\nPlaces with semicolons (likely notes): {len(issues['semicolons'])}")
        for place_id, name in issues['semicolons'][:10]:
            print(f"  {place_id}: {name}")
        if len(issues['semicolons']) > 10:
            print(f"  ... and {len(issues['semicolons']) - 10} more")

        print(f"\nPlaces with ' of ' (pet peeve): {len(issues['lowercase_of'])}")
        for place_id, name in issues['lowercase_of'][:10]:
            print(f"  {place_id}: {name}")
        if len(issues['lowercase_of']) > 10:
            print(f"  ... and {len(issues['lowercase_of']) - 10} more")

        print(f"\nVery long place names (>100 chars): {len(issues['very_long'])}")
        for place_id, name in issues['very_long'][:5]:
            print(f"  {place_id}: {name[:80]}...")
        if len(issues['very_long']) > 5:
            print(f"  ... and {len(issues['very_long']) - 5} more")

        print(f"\nPlaces with 'ruler': {len(issues['contains_ruler'])}")
        for place_id, name in issues['contains_ruler'][:10]:
            print(f"  {place_id}: {name}")

        print(f"\nPlaces with 'see': {len(issues['contains_see'])}")
        for place_id, name in issues['contains_see'][:10]:
            print(f"  {place_id}: {name}")

        print(f"\nDuplicate place names (case-insensitive): {len(issues['duplicates'])}")
        for normalized, place_list in list(issues['duplicates'].items())[:10]:
            print(f"\n  '{normalized}' appears {len(place_list)} times:")
            for place_id, name in place_list[:5]:
                print(f"    PlaceID {place_id}: {name}")
        if len(issues['duplicates']) > 10:
            print(f"  ... and {len(issues['duplicates']) - 10} more duplicate sets")

        # Summary statistics
        print("\n=== SUMMARY ===")
        total_issues = (
            len(issues['parentheses']) +
            len(issues['semicolons']) +
            len(issues['lowercase_of']) +
            len(issues['very_long']) +
            len(issues['contains_ruler']) +
            len(issues['contains_see']) +
            len(issues['duplicates'])
        )
        print(f"Total places with issues: {total_issues}")
        print(f"Total unique places: {len(places)}")
        print(f"Percentage with issues: {total_issues / len(places) * 100:.1f}%")

        # Check how many events reference places
        cursor.execute("SELECT COUNT(*) FROM EventTable WHERE PlaceID IS NOT NULL")
        events_with_places = cursor.fetchone()[0]
        print(f"\nEvents referencing places: {events_with_places:,}")

        return issues


if __name__ == '__main__':
    analyze_places()
