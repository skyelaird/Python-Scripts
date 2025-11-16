#!/usr/bin/env python3
"""
Fix genealogy record for Margaret Verch Duptory (@I25168@):
1. Change title from prefix (NPFX) to suffix (TITL)
2. Remove duplicate FAMS entries (keep only @F15360@, remove @F16192@, @F16202@, @F16207@)
3. Remove duplicate FAMC entries (keep only @F16218@, remove @F17105@, @F17113@, @F17122@)
4. Delete the duplicate family records
"""

import re
from pathlib import Path


def fix_margaret_record():
    gedcom_path = Path("/home/user/Python-Scripts/GedMerge/GEDCOM/Joel2020.ged")

    # Read the entire file
    with open(gedcom_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Track changes
    changes = []
    in_margaret_record = False
    in_duplicate_family = False
    current_family_id = None
    lines_to_delete = set()

    # Family records to delete
    duplicate_fams = ['@F16192@', '@F16202@', '@F16207@']
    duplicate_famc = ['@F17105@', '@F17113@', '@F17122@']
    all_duplicates = duplicate_fams + duplicate_famc

    # First pass: identify lines to delete and modify
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip('\n\r')

        # Check if we're entering Margaret's record
        if stripped == '0 @I25168@ INDI':
            in_margaret_record = True
            changes.append(f"Found Margaret's record at line {i+1}")

        # Check if we're leaving an individual record
        if in_margaret_record and stripped.startswith('0 ') and '@I25168@' not in stripped:
            in_margaret_record = False

        # Within Margaret's record
        if in_margaret_record:
            # Fix title: change NPFX to TITL
            if '2 NPFX Princess Of Ireland' in stripped:
                lines[i] = stripped.replace('2 NPFX Princess Of Ireland', '2 TITL Princess Of Ireland') + '\n'
                changes.append(f"Line {i+1}: Changed NPFX to TITL")

            # Remove duplicate FAMS entries
            elif any(f'1 FAMS {fam_id}' in stripped for fam_id in duplicate_fams):
                lines_to_delete.add(i)
                fam_id = stripped.split()[2]
                changes.append(f"Line {i+1}: Marking duplicate FAMS {fam_id} for deletion")

            # Remove duplicate FAMC entries
            elif any(f'1 FAMC {fam_id}' in stripped for fam_id in duplicate_famc):
                lines_to_delete.add(i)
                fam_id = stripped.split()[2]
                changes.append(f"Line {i+1}: Marking duplicate FAMC {fam_id} for deletion")

        # Check if we're entering a duplicate family record
        if stripped.startswith('0 ') and any(fam_id in stripped for fam_id in all_duplicates):
            in_duplicate_family = True
            current_family_id = stripped.split()[1]
            changes.append(f"Found duplicate family record {current_family_id} at line {i+1}")

        # Mark all lines of duplicate family records for deletion
        if in_duplicate_family:
            lines_to_delete.add(i)
            # Check if we're leaving this family record
            if i + 1 < len(lines) and lines[i + 1].startswith('0 '):
                in_duplicate_family = False
                current_family_id = None

        i += 1

    # Second pass: write output file, excluding deleted lines
    output_lines = [lines[i] for i in range(len(lines)) if i not in lines_to_delete]

    # Create backup
    backup_path = gedcom_path.with_suffix('.ged.backup_margaret')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    changes.append(f"Created backup at {backup_path}")

    # Write modified file
    with open(gedcom_path, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

    # Summary
    print("=" * 70)
    print("MARGARET VERCH DUPTORY RECORD FIXES")
    print("=" * 70)
    print(f"\nTotal lines deleted: {len(lines_to_delete)}")
    print(f"Original file size: {len(lines)} lines")
    print(f"New file size: {len(output_lines)} lines")
    print(f"\nChanges made:")
    for change in changes:
        print(f"  - {change}")
    print(f"\nBackup saved to: {backup_path}")
    print("=" * 70)


if __name__ == '__main__':
    fix_margaret_record()
