"""Database adapter for RootsMagic SQLite databases."""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

from .models import RMPerson, RMFamily, RMEvent, RMName, RMPlace, RMSource, RMCitation
from ..core.place import Place


class RootsMagicDatabase:
    """Adapter for working with RootsMagic .rmtree SQLite databases.

    This class handles:
    - SQLite connection management
    - Custom RMNOCASE collation registration
    - CRUD operations for persons, families, events, etc.
    - Transaction management
    """

    def __init__(self, db_path: str | Path):
        """Initialize database connection.

        Args:
            db_path: Path to the .rmtree database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Access columns by name

        # Register custom collation for RootsMagic
        self._register_rmnocase_collation()

    def _register_rmnocase_collation(self):
        """Register the RMNOCASE collation used by RootsMagic."""
        def rmnocase_collation(s1: str, s2: str) -> int:
            """Case-insensitive collation for RootsMagic."""
            s1_upper = s1.upper() if s1 else ''
            s2_upper = s2.upper() if s2 else ''
            return (s1_upper > s2_upper) - (s1_upper < s2_upper)

        self.conn.create_collation("RMNOCASE", rmnocase_collation)

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    # ========== Statistics Methods ==========

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics.

        Returns:
            Dictionary with counts of various record types
        """
        cursor = self.conn.cursor()
        stats = {}

        tables = [
            ('persons', 'PersonTable'),
            ('families', 'FamilyTable'),
            ('events', 'EventTable'),
            ('names', 'NameTable'),
            ('places', 'PlaceTable'),
            ('sources', 'SourceTable'),
            ('citations', 'CitationTable'),
            ('multimedia', 'MultimediaTable'),
        ]

        for name, table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[name] = cursor.fetchone()[0]

        return stats

    # ========== Person Methods ==========

    def get_person(self, person_id: int, load_names: bool = True,
                   load_events: bool = True) -> Optional[RMPerson]:
        """Get a person by ID.

        Args:
            person_id: The PersonID to retrieve
            load_names: Whether to load associated names
            load_events: Whether to load associated events

        Returns:
            RMPerson object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM PersonTable WHERE PersonID = ?
        """, (person_id,))

        row = cursor.fetchone()
        if not row:
            return None

        person = self._row_to_person(row)

        if load_names:
            person.names = self.get_person_names(person_id)

        if load_events:
            person.events = self.get_person_events(person_id)

        return person

    def get_all_persons(self, limit: Optional[int] = None,
                       offset: int = 0) -> List[RMPerson]:
        """Get all persons from the database.

        Args:
            limit: Maximum number of persons to return
            offset: Number of persons to skip

        Returns:
            List of RMPerson objects
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM PersonTable ORDER BY PersonID"
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        cursor.execute(query)
        persons = [self._row_to_person(row) for row in cursor.fetchall()]

        # Load names for all persons
        for person in persons:
            person.names = self.get_person_names(person.person_id)

        return persons

    def get_person_names(self, person_id: int) -> List[RMName]:
        """Get all names for a person.

        Args:
            person_id: The PersonID

        Returns:
            List of RMName objects
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM NameTable
            WHERE OwnerID = ?
            ORDER BY IsPrimary DESC, NameID
        """, (person_id,))

        return [self._row_to_name(row) for row in cursor.fetchall()]

    def get_person_events(self, person_id: int) -> List[RMEvent]:
        """Get all events for a person.

        Args:
            person_id: The PersonID

        Returns:
            List of RMEvent objects
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM EventTable
            WHERE OwnerType = 0 AND OwnerID = ?
            ORDER BY SortDate, EventID
        """, (person_id,))

        return [self._row_to_event(row) for row in cursor.fetchall()]

    def search_persons_by_name(self, surname: Optional[str] = None,
                              given: Optional[str] = None,
                              fuzzy: bool = False) -> List[RMPerson]:
        """Search for persons by name.

        Args:
            surname: Surname to search for
            given: Given name to search for
            fuzzy: Use LIKE for fuzzy matching

        Returns:
            List of matching RMPerson objects
        """
        cursor = self.conn.cursor()

        conditions = []
        params = []

        if surname:
            if fuzzy:
                conditions.append("Surname LIKE ?")
                params.append(f"%{surname}%")
            else:
                conditions.append("Surname = ?")
                params.append(surname)

        if given:
            if fuzzy:
                conditions.append("Given LIKE ?")
                params.append(f"%{given}%")
            else:
                conditions.append("Given = ?")
                params.append(given)

        if not conditions:
            return []

        where_clause = " AND ".join(conditions)

        cursor.execute(f"""
            SELECT DISTINCT p.*
            FROM PersonTable p
            JOIN NameTable n ON p.PersonID = n.OwnerID
            WHERE {where_clause}
            ORDER BY p.PersonID
        """, params)

        persons = [self._row_to_person(row) for row in cursor.fetchall()]

        # Load names for each person
        for person in persons:
            person.names = self.get_person_names(person.person_id)

        return persons

    # ========== Family Methods ==========

    def get_family(self, family_id: int) -> Optional[RMFamily]:
        """Get a family by ID.

        Args:
            family_id: The FamilyID to retrieve

        Returns:
            RMFamily object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM FamilyTable WHERE FamilyID = ?", (family_id,))

        row = cursor.fetchone()
        return self._row_to_family(row) if row else None

    def get_person_families_as_spouse(self, person_id: int) -> List[RMFamily]:
        """Get families where person is a spouse.

        Args:
            person_id: The PersonID

        Returns:
            List of RMFamily objects
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM FamilyTable
            WHERE FatherID = ? OR MotherID = ?
            ORDER BY FamilyID
        """, (person_id, person_id))

        return [self._row_to_family(row) for row in cursor.fetchall()]

    def get_person_families_as_child(self, person_id: int) -> List[RMFamily]:
        """Get families where person is a child.

        Args:
            person_id: The PersonID

        Returns:
            List of RMFamily objects
        """
        cursor = self.conn.cursor()

        # Need to check ChildTable for children
        cursor.execute("""
            SELECT f.* FROM FamilyTable f
            JOIN ChildTable c ON f.FamilyID = c.FamilyID
            WHERE c.ChildID = ?
            ORDER BY f.FamilyID
        """, (person_id,))

        return [self._row_to_family(row) for row in cursor.fetchall()]

    # ========== Place Methods ==========

    def get_place(self, place_id: int) -> Optional[RMPlace]:
        """Get a place by ID.

        Args:
            place_id: The PlaceID to retrieve

        Returns:
            RMPlace object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PlaceTable WHERE PlaceID = ?", (place_id,))

        row = cursor.fetchone()
        return self._row_to_place(row) if row else None

    # ========== Helper Methods ==========

    def _row_to_person(self, row: sqlite3.Row) -> RMPerson:
        """Convert database row to RMPerson object."""
        return RMPerson(
            person_id=row['PersonID'],
            unique_id=row['UniqueID'],
            sex=row['Sex'],
            parent_id=row['ParentID'],
            spouse_id=row['SpouseID'],
            color=row['Color'],
            color1=row['Color1'],
            color2=row['Color2'],
            color3=row['Color3'],
            color4=row['Color4'],
            color5=row['Color5'],
            color6=row['Color6'],
            color7=row['Color7'],
            color8=row['Color8'],
            color9=row['Color9'],
            relate1=row['Relate1'],
            relate2=row['Relate2'],
            flags=row['Flags'],
            living=bool(row['Living']),
            is_private=bool(row['IsPrivate']),
            proof=row['Proof'],
            bookmark=bool(row['Bookmark']),
            note=row['Note'],
            utc_mod_date=row['UTCModDate'],
        )

    def _row_to_name(self, row: sqlite3.Row) -> RMName:
        """Convert database row to RMName object."""
        return RMName(
            name_id=row['NameID'],
            owner_id=row['OwnerID'],
            surname=row['Surname'],
            given=row['Given'],
            prefix=row['Prefix'],
            suffix=row['Suffix'],
            nickname=row['Nickname'],
            name_type=row['NameType'],
            date=row['Date'],
            sort_date=row['SortDate'],
            is_primary=bool(row['IsPrimary']),
            is_private=bool(row['IsPrivate']),
            proof=row['Proof'],
            sentence=row['Sentence'],
            note=row['Note'],
            birth_year=row['BirthYear'],
            death_year=row['DeathYear'],
            display=row['Display'],
            language=row['Language'],
            utc_mod_date=row['UTCModDate'],
            surname_mp=row['SurnameMP'],
            given_mp=row['GivenMP'],
            nickname_mp=row['NicknameMP'],
        )

    def _row_to_event(self, row: sqlite3.Row) -> RMEvent:
        """Convert database row to RMEvent object."""
        return RMEvent(
            event_id=row['EventID'],
            event_type=row['EventType'],
            owner_type=row['OwnerType'],
            owner_id=row['OwnerID'],
            family_id=row['FamilyID'],
            place_id=row['PlaceID'],
            site_id=row['SiteID'],
            date=row['Date'],
            sort_date=row['SortDate'],
            is_primary=bool(row['IsPrimary']),
            is_private=bool(row['IsPrivate']),
            proof=row['Proof'],
            status=row['Status'],
            sentence=row['Sentence'],
            details=row['Details'],
            note=row['Note'],
            utc_mod_date=row['UTCModDate'],
        )

    def _row_to_family(self, row: sqlite3.Row) -> RMFamily:
        """Convert database row to RMFamily object."""
        return RMFamily(
            family_id=row['FamilyID'],
            father_id=row['FatherID'],
            mother_id=row['MotherID'],
            child_id=row['ChildID'],
            husb_order=row['HusbOrder'],
            wife_order=row['WifeOrder'],
            is_private=bool(row['IsPrivate']),
            proof=row['Proof'],
            spouse_label=row['SpouseLabel'],
            father_label=row['FatherLabel'],
            mother_label=row['MotherLabel'],
            spouse_label_str=row['SpouseLabelStr'],
            father_label_str=row['FatherLabelStr'],
            mother_label_str=row['MotherLabelStr'],
            note=row['Note'],
            utc_mod_date=row['UTCModDate'],
        )

    def _row_to_place(self, row: sqlite3.Row) -> RMPlace:
        """Convert database row to RMPlace object."""
        # Check if multilingual_names column exists
        multilingual_names = None
        if 'MultilingualNames' in row.keys():
            multilingual_names = row['MultilingualNames']

        return RMPlace(
            place_id=row['PlaceID'],
            place_type=row['PlaceType'],
            name=row['Name'],
            abbrev=row['Abbrev'],
            normalized=row['Normalized'],
            latitude=row['Latitude'],
            longitude=row['Longitude'],
            lat_long_exact=bool(row['LatLongExact']),
            master_id=row['MasterID'],
            note=row['Note'],
            reverse=row['Reverse'],
            fs_id=row['fsID'],
            an_id=row['anID'],
            utc_mod_date=row['UTCModDate'],
            multilingual_names=multilingual_names,
        )

    def rm_place_to_place(self, rm_place: RMPlace) -> Place:
        """Convert RMPlace to generic Place object.

        Args:
            rm_place: RMPlace database object

        Returns:
            Place object with multilingual support
        """
        names = rm_place.get_multilingual_names_dict()

        # Determine primary language (default to 'en' if not specified)
        primary_language = 'en'

        return Place(
            names=names,
            primary_language=primary_language,
            latitude=rm_place.get_latitude_decimal(),
            longitude=rm_place.get_longitude_decimal(),
            place_type='database',
            notes=rm_place.note,
            hierarchy=[]
        )

    def place_to_rm_place(self, place: Place, place_id: int = 0) -> RMPlace:
        """Convert generic Place object to RMPlace.

        Args:
            place: Generic Place object
            place_id: The place ID (0 for new places)

        Returns:
            RMPlace database object
        """
        names = place.get_all_names()
        primary_name = place.get_name()

        rm_place = RMPlace(place_id=place_id)
        rm_place.set_multilingual_names_dict(names)
        rm_place.name = primary_name

        if place.latitude is not None and place.longitude is not None:
            rm_place.set_coordinates_decimal(place.latitude, place.longitude)

        rm_place.note = place.notes

        return rm_place

    def ensure_multilingual_names_column(self) -> bool:
        """Ensure the PlaceTable has a MultilingualNames column.

        This adds the column if it doesn't exist. Safe to call multiple times.

        Returns:
            True if column was added, False if it already existed
        """
        cursor = self.conn.cursor()

        # Check if column exists
        cursor.execute("PRAGMA table_info(PlaceTable)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'MultilingualNames' not in columns:
            # Add the column
            cursor.execute("""
                ALTER TABLE PlaceTable
                ADD COLUMN MultilingualNames TEXT
            """)
            self.conn.commit()
            return True

        return False
