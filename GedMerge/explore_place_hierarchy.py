#!/usr/bin/env python3
"""Explore RootsMagic place hierarchy and structure."""

from gedmerge.rootsmagic import RootsMagicDatabase
from pathlib import Path
import zipfile
import re
from collections import Counter


def extract_db():
    """Extract database if needed."""
    zip_path = Path('/home/user/Python-Scripts/GedMerge/Rootsmagic/Joel2020.zip')
    extract_dir = zip_path.parent
    db_path = extract_dir / 'Joel2020.rmtree'

    if not db_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    return db_path


def explore_hierarchy():
    """Explore place hierarchy using MasterID and PlaceType."""
    db_path = extract_db()

    with RootsMagicDatabase(db_path) as db:
        cursor = db.conn.cursor()

        # Get all places with their full details
        cursor.execute("""
            SELECT PlaceID, PlaceType, Name, Normalized, MasterID, Abbrev
            FROM PlaceTable
            ORDER BY PlaceID
        """)
        places = cursor.fetchall()

        print(f"=== EXPLORING {len(places)} PLACES ===\n")

        # Analyze PlaceType distribution
        print("=== PLACE TYPE DISTRIBUTION ===")
        place_types = Counter(p[1] for p in places)
        for ptype, count in sorted(place_types.items()):
            print(f"  PlaceType {ptype}: {count:,} places")

        # Analyze MasterID usage
        print("\n=== MASTER ID USAGE ===")
        with_master = [p for p in places if p[4] is not None]
        without_master = [p for p in places if p[4] is None]
        print(f"Places with MasterID: {len(with_master):,}")
        print(f"Places without MasterID: {len(without_master):,}")

        # Show examples of master-detail relationships
        if with_master:
            print("\n=== MASTER-DETAIL RELATIONSHIP EXAMPLES ===")
            for i, place in enumerate(with_master[:10]):
                place_id, place_type, name, normalized, master_id, abbrev = place

                # Get the master place
                cursor.execute("""
                    SELECT PlaceID, PlaceType, Name FROM PlaceTable WHERE PlaceID = ?
                """, (master_id,))
                master = cursor.fetchone()

                if master:
                    master_id, master_type, master_name = master
                    print(f"\nDetail Place: {name}")
                    print(f"  PlaceID: {place_id}, Type: {place_type}")
                    print(f"  Master: {master_name}")
                    print(f"  MasterID: {master_id}, Type: {master_type}")

        # Look for postal codes
        print("\n=== POSTAL CODE ANALYSIS ===")

        # French postal codes (5 digits)
        french_postal = []
        # Canadian postal codes (A1A 1A1)
        canadian_postal = []
        # UK postal codes
        uk_postal = []
        # Other number patterns
        other_numbers = []

        for place in places:
            place_id, place_type, name, normalized, master_id, abbrev = place
            if not name:
                continue

            # French postal code pattern (5 digits)
            if re.search(r'\b\d{5}\b', name) and ('france' in name.lower() or 'paris' in name.lower()):
                french_postal.append((place_id, name))

            # Canadian postal code pattern
            elif re.search(r'\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b', name, re.IGNORECASE):
                canadian_postal.append((place_id, name))

            # UK postal code pattern
            elif re.search(r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}\b', name) and 'england' in name.lower():
                uk_postal.append((place_id, name))

            # Any place that's mostly numbers
            elif re.search(r'^\d+$', name.strip()):
                other_numbers.append((place_id, name))

        print(f"\nFrench postal codes found: {len(french_postal)}")
        for place_id, name in french_postal[:10]:
            print(f"  {place_id}: {name}")
        if len(french_postal) > 10:
            print(f"  ... and {len(french_postal) - 10} more")

        print(f"\nCanadian postal codes found: {len(canadian_postal)}")
        for place_id, name in canadian_postal[:10]:
            print(f"  {place_id}: {name}")

        print(f"\nUK postal codes found: {len(uk_postal)}")
        for place_id, name in uk_postal[:10]:
            print(f"  {place_id}: {name}")

        print(f"\nPlaces that are just numbers: {len(other_numbers)}")
        for place_id, name in other_numbers[:10]:
            print(f"  {place_id}: {name}")

        # Look for University of Alberta specifically
        print("\n=== SEARCHING FOR 'UNIVERSITY OF ALBERTA' ===")
        cursor.execute("""
            SELECT PlaceID, PlaceType, Name, MasterID
            FROM PlaceTable
            WHERE Name LIKE '%university of alberta%'
            COLLATE NOCASE
        """)
        uofa_places = cursor.fetchall()

        for place_id, place_type, name, master_id in uofa_places:
            print(f"\nPlaceID {place_id}: {name}")
            print(f"  Type: {place_type}, MasterID: {master_id}")

            if master_id:
                cursor.execute("SELECT Name FROM PlaceTable WHERE PlaceID = ?", (master_id,))
                master = cursor.fetchone()
                if master:
                    print(f"  Master place: {master[0]}")

            # Check how many events use this place
            cursor.execute("SELECT COUNT(*) FROM EventTable WHERE PlaceID = ?", (place_id,))
            event_count = cursor.fetchone()[0]
            print(f"  Used in {event_count} events")

        # Look for other institutional/specific locations
        print("\n=== INSTITUTIONAL/SPECIFIC LOCATIONS ===")
        institutions = [
            'university', 'hospital', 'church', 'cemetery', 'school',
            'college', 'abbey', 'priory', 'cathedral', 'chapel'
        ]

        inst_places = {}
        for inst in institutions:
            cursor.execute(f"""
                SELECT COUNT(*)
                FROM PlaceTable
                WHERE Name LIKE '%{inst}%'
                COLLATE NOCASE
            """)
            count = cursor.fetchone()[0]
            inst_places[inst] = count

        print("\nInstitutional location counts:")
        for inst, count in sorted(inst_places.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {inst}: {count:,}")

        # Check events to see if detailed places are actually useful
        print("\n=== USAGE ANALYSIS ===")
        cursor.execute("""
            SELECT COUNT(DISTINCT PlaceID)
            FROM EventTable
            WHERE PlaceID IS NOT NULL
        """)
        used_places = cursor.fetchone()[0]

        print(f"Total places defined: {len(places):,}")
        print(f"Places actually used in events: {used_places:,}")
        print(f"Unused places: {len(places) - used_places:,}")

        # Find unused postal codes
        used_postal_ids = set()
        cursor.execute("SELECT DISTINCT PlaceID FROM EventTable WHERE PlaceID IS NOT NULL")
        for row in cursor.fetchall():
            used_postal_ids.add(row[0])

        unused_french = [p for p in french_postal if p[0] not in used_postal_ids]
        print(f"\nUnused French postal codes: {len(unused_french)} of {len(french_postal)}")


if __name__ == '__main__':
    explore_hierarchy()
