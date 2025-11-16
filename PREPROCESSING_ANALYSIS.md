# GedMerge Preprocessing Workflow Analysis

## Executive Summary

This genealogy project (GedMerge) implements a sophisticated **3-phase preprocessing workflow** designed to standardize genealogy names before duplicate detection. The current implementation has **2 of 3 phases completed** with proper infrastructure for the third phase (duplicate detection/merging).

**Current Status**: Phase 1 & 2 fully implemented; Phase 3 (duplicate detection) is a planned feature.

---

## 1. PREPROCESSING STEPS CURRENTLY IMPLEMENTED

### Phase 1: Structural Cleanup (COMPLETE)
**Tool**: `preprocess_names_for_matching.py`

This script applies genealogy naming conventions and standardizes name structures:

#### 1.1 NN Convention for Missing Given Names
- **What**: Replaces NULL/empty given names with "NN" (No Name)
- **Why**: Genealogy standard; prevents database NULL issues; enables proper string comparisons
- **Code location**: Lines 172-178 in preprocess_names_for_matching.py

#### 1.2 Embedded Language Variant Detection
- **What**: Identifies names with bracketed/parenthetical variants like "Margaret [Marguerite]"
- **Why**: These should be separate records with different language codes, not combined
- **Code location**: Lines 180-228 in preprocess_names_for_matching.py
- **Patterns detected**:
  ```
  Margaret [Marguerite]  → Create separate name records
  Wilhelm (William)      → Flag for review if it looks like variant
  ```

#### 1.3 Placeholder Surname Removal
- **What**: Removes meaningless placeholder surnames
- **Patterns removed**:
  ```
  EndofLine      → Generic placeholder
  Unknown        → No information
  ???            → Question marks
  ---            → Dashes
  ```
- **Code location**: Lines 230-242 in preprocess_names_for_matching.py
- **Important**: Preserves meaningful surname differences (e.g., mother's maiden name)

#### 1.4 Language Code Detection
- **What**: Detects language from name patterns and sets ISO 639-1 codes
- **Supported languages**: en, fr, de, it, es, la
- **Detection method**: Pattern matching (French: Jean, Marie; German: Wilhelm, Heinrich, etc.)
- **Code location**: Lines 247-300 in preprocess_names_for_matching.py
- **Safe for RootsMagic**: Language field is standard in NameTable

#### 1.5 MRS Placeholder Detection
- **What**: Identifies unnamed entries marked as "MRS", "Ms.", "Miss"
- **Why**: These are often placeholders for mothers/wives without proper names
- **Code location**: Lines 194-200, 301-306 in preprocess_names_for_matching.py

#### Execution Options
```bash
# Dry-run mode (default - see what would change)
python preprocess_names_for_matching.py database.rmtree --report

# Execute changes
python preprocess_names_for_matching.py database.rmtree --execute
```

#### Statistics Generated
- By change type (nn_convention, embedded_variant, placeholder_surname, etc.)
- Detailed report showing person context, old/new values

---

### Phase 2: Language Analysis and Variant Separation (COMPLETE)
**Tool**: `analyze_name_structure.py`

Identifies structural and language issues, then fixes embedded variants:

#### 2.1 Structural Issue Detection
- **Missing given name**: Primary name has only surname
- **Missing surname**: Primary name has only given name  
- **Reversed names**: Names stored as "Smith, John" instead of proper GEDCOM format
- **Titles in wrong fields**: "Sir William" in Given field instead of Prefix field
- **Code location**: Lines 200-248 in analyze_name_structure.py

#### 2.2 Language Issue Detection
- **Missing language codes**: Alternate names without language codes
- **Language mismatch**: Name pattern suggests one language but different code set
- **French/German detection**: Pattern matching for common names
- **Code location**: Lines 250-304 in analyze_name_structure.py

#### 2.3 Embedded Variant Fixing (Dry-run or Execute)
- **What**: Creates separate name records for bracketed variants
- **Process**:
  1. Find name with "Margaret [Marguerite]"
  2. Keep "Margaret" as primary
  3. Create new alternate record with "Marguerite"
  4. Set appropriate language codes
- **Code location**: Lines 443-530 in analyze_name_structure.py

#### 2.4 Issue Severity Levels
- **High**: Embedded variants, missing given name in primary record
- **Medium**: Embedded variants in parentheses, possible reversed names, missing surnames
- **Low**: Missing language codes, alternate names without language

#### Execution Options
```bash
# Analyze and show summary
python analyze_name_structure.py database.rmtree

# Show detailed report for specific issue
python analyze_name_structure.py database.rmtree --detail embedded_variant_bracket

# Fix embedded variants (dry-run)
python analyze_name_structure.py database.rmtree --fix-variants --dry-run

# Actually fix variants
python analyze_name_structure.py database.rmtree --fix-variants --execute
```

#### Reports Generated
- Summary of all issues found (total count, breakdown by type)
- Severity breakdown (high/medium/low)
- Detailed report showing exact issues and suggested fixes

---

### Phase 3: Cleanup Unnamed People (ADDITIONAL PREPROCESSING TOOL)
**Tool**: `cleanup_unnamed_people.py`

Removes truly anonymous placeholder records that don't contribute genealogically:

#### 3.1 Deletion Criteria
A person is marked for deletion if they meet one of these conditions:

1. **EndofLine parent**: Has only surname matching their child's surname
   ```
   "Smith" as parent of "John Smith" → DELETE
   ```

2. **MRS placeholder without named ancestors**: Generic titles with no genealogical value
   ```
   "Mrs." / "Ms." / "Miss" AND no named ancestors above → DELETE
   ```

3. **Unnamed without named ancestors**: No given name AND no traceable ancestry
   ```
   "" (empty) or no real name AND no named parents → DELETE
   ```

#### 3.2 Merge Candidates
- **MRS persons with potential matches**: Flagged for manual review, NOT automatically deleted
- Example: "Mrs. Jane Smith" might be a duplicate of "Jane Williams" elsewhere
- **Code location**: Lines 101-123 in cleanup_unnamed_people.py

#### 3.3 Safety Features
- **Dry-run by default**: No deletions without `--execute` flag
- **Confirmation prompt**: Requires explicit "yes" before execution
- **Transaction safety**: All deletions in single transaction (rollback on error)
- **Detailed reporting**: Shows exactly what will be deleted and why

#### Execution Options
```bash
# Dry-run (preview)
python cleanup_unnamed_people.py database.rmtree --dry-run

# Show detailed report (up to 100 entries)
python cleanup_unnamed_people.py database.rmtree --dry-run --report-limit 100

# Execute deletions (after backup!)
cp database.rmtree database.rmtree.backup
python cleanup_unnamed_people.py database.rmtree --execute
```

#### What Gets Deleted
For each person deleted:
- Name records (NameTable)
- Event records (EventTable)
- Family connections (FatherID/MotherID set to NULL)
- Child relationships (ChildTable)
- Person record (PersonTable)

---

## 2. EXISTING DUPLICATE DETECTION/MERGING CODE

### Current Status: NOT YET IMPLEMENTED

The infrastructure is in place but the core matching logic is planned for Phase 2/3:

#### What Exists:
```
GedMerge/
├── matching/          # Placeholder module for future matching algorithms
│   └── __init__.py    # Currently empty
└── merge/             # Placeholder module for future merge strategies
    └── __init__.py    # Currently empty
```

#### CLI Placeholders
In `gedmerge/ui/cli.py`:
```python
elif args.command == 'find-duplicates':
    print("Find duplicates functionality coming in Phase 2!")
    return 0
elif args.command == 'merge':
    print("Merge functionality coming in Phase 3!")
    return 0
```

#### Proposed Matching Strategy (from NAMING_CONVENTIONS.md)
```
1. Metaphone matching (surname_mp, given_mp)
2. Fuzzy string matching (rapidfuzz)
3. Date proximity (birth/death within N years)
4. Place proximity (same location or nearby)
5. Relationship analysis (same parents/spouse/children)
6. Scoring system (confidence level)
```

#### Dependencies Already Available
- **rapidfuzz**: Fast fuzzy string matching (in dependencies)
- **phonetics**: Phonetic algorithms (Soundex, Metaphone, etc.)
- **python-gedcom**: GEDCOM file parsing
- **RootsMagic adapter**: Direct database access

---

## 3. OVERALL WORKFLOW AND DATA FLOW

### Complete Preprocessing Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                       RAW GENEALOGY DATA                         │
│           (GEDCOM file or RootsMagic database)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   PHASE 1: STRUCTURAL CLEANUP    │
        │  preprocess_names_for_matching   │
        └──────────────────────────────────┘
                  │
                  ├─→ Apply NN convention
                  │   (missing given names)
                  │
                  ├─→ Remove placeholder surnames
                  │   (EndofLine, Unknown, etc.)
                  │
                  ├─→ Detect embedded variants
                  │   (Margaret [Marguerite])
                  │
                  ├─→ Detect language patterns
                  │
                  └─→ Flag MRS placeholders
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   PHASE 2: LANGUAGE ANALYSIS     │
        │    analyze_name_structure        │
        └──────────────────────────────────┘
                  │
                  ├─→ Set language codes
                  │
                  ├─→ Create separate name
                  │   records for variants
                  │
                  ├─→ Fix structural issues
                  │   (reversed names, titles)
                  │
                  └─→ Validate name structure
                           │
                           ▼
        ┌──────────────────────────────────┐
        │   PHASE 3: CLEANUP UNNAMED       │
        │  cleanup_unnamed_people          │
        └──────────────────────────────────┘
                  │
                  ├─→ Identify EndofLine parents
                  │
                  ├─→ Find MRS placeholders
                  │   without ancestors
                  │
                  ├─→ Find unnamed without
                  │   genealogical value
                  │
                  ├─→ Flag merge candidates
                  │   (potential duplicates)
                  │
                  └─→ Delete anonymous records
                           │
                           ▼
    ┌─────────────────────────────────────┐
    │  CLEAN, STANDARDIZED DATA READY FOR │
    │         DUPLICATE DETECTION         │
    └─────────────────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────────────────┐
    │   [PLANNED] PHASE 4: DUPLICATE      │
    │   DETECTION & MERGING               │
    │   - Find duplicates (metaphone,     │
    │     fuzzy matching, relationships)  │
    │   - Merge with conflict resolution  │
    │   - Preserve all data               │
    └─────────────────────────────────────┘
```

### Data Flow Architecture

#### 1. Data Input Sources
- **GEDCOM files**: Via `GedcomParser.load_gedcom()`
- **RootsMagic databases**: Via `RootsMagicDatabase` adapter
- **Database schema**: PersonTable, NameTable, FamilyTable, EventTable, PlaceTable

#### 2. Data Processing Pipeline
```
Raw Data
   ↓
GedcomParser / RootsMagicDatabase (Load)
   ↓
NamePreprocessor (Phase 1: Structural cleanup)
   ↓
NameStructureAnalyzer (Phase 2: Language analysis)
   ↓
PersonCleaner (Phase 3: Remove unnamed)
   ↓
[Future] DuplicateMatcher (Phase 4: Find duplicates)
   ↓
[Future] DuplicateMerger (Phase 5: Merge with conflict resolution)
   ↓
GedcomParser.save_gedcom() (Output merged GEDCOM)
```

#### 3. Core Data Models
- **Person**: Name, sex, events (birth, death, burial), family relationships
- **Name**: Given, surname, prefix, suffix, language code, type (primary/alternate)
- **Event**: Type, date, place, notes (for birth, death, marriage, etc.)
- **Place**: Multilingual support (en, fr, de, etc.), coordinates
- **Family**: Husband, wife, children, marriage event

#### 4. Key Database Tables Used
```
PersonTable          → Person records
NameTable           → Name variants (with Language field!)
FamilyTable         → Family units
ChildTable          → Parent-child relationships
EventTable          → Life events
PlaceTable          → Geographic locations
```

---

## 4. WHAT COMES AFTER PREPROCESSING

### Phase 4: Duplicate Detection (PLANNED)

Once preprocessing is complete, the system will:

#### 4.1 Matching Algorithm (Conceptual, not yet implemented)
```python
def find_duplicates(person1, person2):
    scores = {}
    
    # 1. Metaphone matching (phonetic)
    scores['surname_phonetic'] = metaphone_compare(p1.surname, p2.surname)
    scores['given_phonetic'] = metaphone_compare(p1.given, p2.given)
    
    # 2. Fuzzy string matching (rapidfuzz)
    scores['surname_fuzzy'] = fuzzy_ratio(p1.surname, p2.surname)
    scores['given_fuzzy'] = fuzzy_ratio(p1.given, p2.given)
    
    # 3. Date proximity
    scores['birth_date'] = date_proximity(p1.birth, p2.birth)
    scores['death_date'] = date_proximity(p1.death, p2.death)
    
    # 4. Place proximity
    scores['birth_place'] = place_proximity(p1.birth_place, p2.birth_place)
    scores['death_place'] = place_proximity(p1.death_place, p2.death_place)
    
    # 5. Relationship analysis
    scores['same_parents'] = parents_match(p1, p2)
    scores['same_spouse'] = spouse_match(p1, p2)
    scores['same_children'] = children_match(p1, p2)
    
    # Calculate composite score
    total_score = weighted_average(scores)
    
    return total_score, confidence_level
```

#### 4.2 Language-Aware Matching
With preprocessing complete:
- "Margaret" [en] vs "Marguerite" [fr] → **Alternate names** (not duplicates)
- "Wilhelm" [de] vs "William" [en] → **Same person, different language variants**
- "NN /Smith/" vs empty → **Consistent handling** (both standardized)

#### 4.3 Scoring System (TBD)
- **High confidence** (90%+): Exact match on surname + birth/death dates match
- **Medium confidence** (70-90%): Phonetic match + date proximity + place match
- **Low confidence** (<70%): Fuzzy match only, requires manual review

#### 4.4 Conflict Resolution in Merging
When merging duplicates, keep:
- **Primary name**: The main person's current name
- **Alternate names**: All other name variants become alternates
- **All events**: Birth, death, marriage dates preserved
- **All sources**: Citations maintained
- **Family relationships**: Consolidated, duplicates removed
- **Notes**: Both versions concatenated with source indication

#### 4.5 Merge Output
- **GEDCOM export**: Standard genealogy format for sharing
- **RootsMagic update**: Direct database modification with audit trail
- **Report**: Summary of duplicates found and merged

---

### Phase 5: Post-Merge Cleanup (FUTURE)

After merging:
1. Remove duplicate family relationships
2. Update cross-references
3. Regenerate statistics
4. Validate data integrity
5. Generate merge report/audit log

---

## 5. SUMMARY: CURRENT STATE VS. ROADMAP

### What's Implemented (READY NOW)
✅ **Phase 1**: Structural name cleanup (NN convention, placeholders, embedded variants)
✅ **Phase 2**: Language analysis and variant separation  
✅ **Phase 3**: Cleanup unnamed/placeholder records
✅ **Core data models**: Person, Family, Event, Place, Name with full support
✅ **RootsMagic database adapter**: Direct SQLite access with proper collations
✅ **GEDCOM parser**: Read/write genealogy files
✅ **CLI interface**: Command-line tools for all preprocessing phases
✅ **Error handling**: Dry-run modes, transaction safety, detailed reporting

### What's Planned (NEXT STEPS)
🔄 **Phase 4**: Duplicate detection algorithms
   - Metaphone matching functions
   - Fuzzy string matching integration
   - Date/place proximity calculations
   - Relationship-based matching
   - Scoring system with confidence levels

🔄 **Phase 5**: Intelligent merging
   - Merge strategy selection
   - Conflict resolution
   - Data preservation validation
   - Audit trail logging

🔄 **Phase 6**: Interactive review UI
   - GUI for reviewing duplicate candidates
   - Approve/reject/merge workflows
   - Batch operations

---

## 6. LOGICAL NEXT STEPS FOR DEVELOPMENT

### Step 1: Implement Metaphone Matching Module
**File**: `gedmerge/matching/phonetic.py`
```python
from phonetics import metaphone, soundex

def metaphone_match(name1: str, name2: str) -> float:
    """Score names on phonetic similarity"""
    m1 = metaphone(name1)
    m2 = metaphone(name2)
    return 1.0 if m1 == m2 else 0.0
```

### Step 2: Implement Fuzzy Matching Module  
**File**: `gedmerge/matching/fuzzy.py`
```python
from rapidfuzz import fuzz

def fuzzy_match(name1: str, name2: str, threshold: float = 0.8) -> float:
    """Score names on string similarity"""
    return fuzz.ratio(name1, name2) / 100.0
```

### Step 3: Implement Date/Place Proximity
**File**: `gedmerge/matching/temporal.py`
```python
def date_proximity(date1: str, date2: str, window_years: int = 5) -> float:
    """Score dates on proximity"""
    # Extract years and compare
```

### Step 4: Implement Relationship Matcher
**File**: `gedmerge/matching/relationships.py`
```python
def same_parents(person1: Person, person2: Person) -> bool:
    """Check if both have same parents"""
```

### Step 5: Implement Scoring System
**File**: `gedmerge/matching/scorer.py`
```python
def calculate_duplicate_score(person1: Person, person2: Person) -> Tuple[float, str]:
    """Calculate confidence score and reason for match"""
```

### Step 6: Implement Find-Duplicates Command
**File**: Update `gedmerge/ui/cli.py`
```python
def find_duplicates_command(args):
    """Execute duplicate detection on GEDCOM/RootsMagic data"""
```

### Step 7: Implement Merge Logic
**File**: `gedmerge/merge/merger.py`
```python
class DuplicateMerger:
    def merge(self, person1: Person, person2: Person) -> Person:
        """Intelligently merge two persons"""
```

---

## Key Success Factors

1. **Preprocessing completeness**: All three phases must complete before duplicate detection
   - This ensures "apples to apples" name comparisons
   - Language codes enable language-aware matching
   - Embedded variants become separate records

2. **Language-aware approach**: Different languages create genuine variance
   - "Margaret" (en) vs "Marguerite" (fr) are alternate forms, NOT duplicates
   - Language codes enable proper deduplication

3. **Conservative deletion approach**: 
   - Only delete records with NO genealogical value
   - Flag merge candidates for manual review
   - Preserve ancestor connections

4. **Audit trail and reversibility**:
   - All changes logged in database
   - Dry-run modes before execution
   - Transaction safety with rollback capability

---

## Files and Line References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Phase 1 Preprocessing | preprocess_names_for_matching.py | 1-472 | Structural cleanup |
| Phase 2 Analysis | analyze_name_structure.py | 1-668 | Language analysis |
| Phase 3 Cleanup | cleanup_unnamed_people.py | 1-494 | Remove placeholders |
| Data Models | gedmerge/core/person.py | 1-204 | Person class |
| | gedmerge/core/family.py | 1-164 | Family class |
| | gedmerge/core/event.py | 1-144 | Event class |
| | gedmerge/core/place.py | (not shown) | Place with multilingual |
| Parser | gedmerge/core/gedcom_parser.py | 1-649 | GEDCOM I/O |
| RootsMagic | gedmerge/rootsmagic/adapter.py | 1-150+ | SQLite database access |
| CLI | gedmerge/ui/cli.py | 1-211 | Command-line interface |
| Documentation | NAMING_CONVENTIONS.md | 1-264 | Full preprocessing guide |
| | CLEANUP_UNNAMED_README.md | 1-191 | Cleanup tool guide |
| | GedMerge/README.md | 1-179 | Project overview |

