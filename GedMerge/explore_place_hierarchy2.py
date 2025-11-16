#!/usr/bin/env python3
"""Deep dive into RootsMagic place hierarchy."""

from gedmerge.rootsmagic import RootsMagicDatabase
from pathlib import Path
import zipfile


def extract_db():
    """Extract database if needed."""
    zip_path = Path('/home/user/Python-Scripts/GedMerge/Rootsmagic/Joel2020.zip')
    extract_dir = zip_path.parent
    db_path = extract_dir / 'Joel2020.rmtree'

    if not db_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    return db_path


def main():
    db_path = extract_db()

    with RootsMagicDatabase(db_path) as db:
        cursor = db.conn.cursor()

        # Check MasterID distribution
        print("=== MASTER ID DISTRIBUTION ===")
        cursor.execute("SELECT COUNT(*) FROM PlaceTable WHERE MasterID = 0")
        no_master = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM PlaceTable WHERE MasterID > 0")
        has_master = cursor.fetchone()[0]

        print(f"Places with MasterID = 0 (no master): {no_master:,}")
        print(f"Places with MasterID > 0 (has master): {has_master:,}")

        # Show examples of places WITH masters
        print("\n=== PLACES WITH MASTER RELATIONSHIPS (MasterID > 0) ===")
        cursor.execute("""
            SELECT p.PlaceID, p.PlaceType, p.Name, p.MasterID, m.Name as MasterName
            FROM PlaceTable p
            LEFT JOIN PlaceTable m ON p.MasterID = m.PlaceID
            WHERE p.MasterID > 0
            ORDER BY p.PlaceID
            LIMIT 20
        """)

        for row in cursor.fetchall():
            place_id, place_type, name, master_id, master_name = row
            print(f"\nDetail: {name}")
            print(f"  PlaceID: {place_id}, Type: {place_type}")
            print(f"  Master: {master_name} (ID: {master_id})")

        # Check PlaceType meanings
        print("\n=== ANALYZING PLACE TYPES ===")
        for ptype in [0, 1, 2]:
            cursor.execute("""
                SELECT PlaceID, Name, MasterID
                FROM PlaceTable
                WHERE PlaceType = ?
                LIMIT 5
            """, (ptype,))

            print(f"\nPlaceType {ptype} examples:")
            for place_id, name, master_id in cursor.fetchall():
                print(f"  {place_id}: {name} (MasterID: {master_id})")

        # Look at churches and cemeteries - often detailed places
        print("\n=== CHURCH EXAMPLES (Likely detail places) ===")
        cursor.execute("""
            SELECT p.PlaceID, p.Name, p.MasterID, m.Name as MasterName
            FROM PlaceTable p
            LEFT JOIN PlaceTable m ON p.MasterID = m.PlaceID
            WHERE p.Name LIKE '%church%'
            COLLATE NOCASE
            LIMIT 10
        """)

        for row in cursor.fetchall():
            place_id, name, master_id, master_name = row
            if master_id and master_id > 0:
                print(f"\n{name}")
                print(f"  Master: {master_name}")
            else:
                print(f"\n{name} (no master)")

        # Check postal codes usage
        print("\n=== POSTAL CODE EXAMPLES ===")
        cursor.execute("""
            SELECT PlaceID, Name, MasterID
            FROM PlaceTable
            WHERE Name LIKE '%35000%'
               OR Name LIKE '%31000%'
               OR Name LIKE '%88200%'
            LIMIT 10
        """)

        for place_id, name, master_id in cursor.fetchall():
            print(f"\n{name}")
            print(f"  PlaceID: {place_id}, MasterID: {master_id}")

            # Check usage
            cursor.execute("SELECT COUNT(*) FROM EventTable WHERE PlaceID = ?", (place_id,))
            usage = cursor.fetchone()[0]
            print(f"  Used in {usage} events")

            # See what events use it
            if usage > 0:
                cursor.execute("""
                    SELECT e.EventType, e.Date
                    FROM EventTable e
                    WHERE e.PlaceID = ?
                    LIMIT 3
                """, (place_id,))
                for event_type, date in cursor.fetchall():
                    event_names = {1: 'Birth', 2: 'Death', 3: 'Burial', 4: 'Marriage'}
                    print(f"    - {event_names.get(event_type, f'Type {event_type}')}: {date}")

        # Find what's in the "of" places
        print("\n=== 'OF' ANALYSIS ===")
        cursor.execute("""
            SELECT Name
            FROM PlaceTable
            WHERE Name LIKE '% of %'
            COLLATE NOCASE
            LIMIT 30
        """)

        legitimate_of = []
        questionable_of = []

        for row in cursor.fetchall():
            name = row[0]
            # Legitimate uses of "of"
            if any(x in name.lower() for x in [
                'district of columbia',
                'isle of ',
                'university of',
                'priory of',
                'abbey of',
                'church of',
                'cathedral of',
                'bay of',
                'cape of',
                'gulf of',
                'sea of',
            ]):
                legitimate_of.append(name)
            else:
                questionable_of.append(name)

        print(f"\nLegitimate 'of' uses (institutional/geographic): {len(legitimate_of)}")
        for name in legitimate_of[:10]:
            print(f"  - {name}")

        print(f"\nQuestionable 'of' uses: {len(questionable_of)}")
        for name in questionable_of[:10]:
            print(f"  - {name}")


if __name__ == '__main__':
    main()
