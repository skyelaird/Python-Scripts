#!/usr/bin/env python3
"""
Batch TIFF Strip Recovery
Process all corrupted TIFFs in a directory
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Import the strip recovery function
exec(open('strip_recovery.py', encoding='utf-8').read(), globals())

def batch_process_directory(source_dir, output_dir):
    """Process all TIFF files in directory"""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read good_files.txt if it exists
    good_files_set = set()
    good_files_path = source_path / "good_files.txt"
    if good_files_path.exists():
        with open(good_files_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.lower().endswith('.tif'):
                    good_files_set.add(line.lower())
        print(f"Loaded {len(good_files_set)} known-good files from good_files.txt")
        print()
    
    # Find all TIFF files (case-insensitive, avoid duplicates)
    tiff_files = []
    seen = set()
    for pattern in ["*.tif", "*.TIF", "*.tiff", "*.TIFF"]:
        for file in source_path.glob(pattern):
            # Use lowercase name as key to avoid case-sensitive duplicates
            key = file.name.lower()
            if key not in seen:
                tiff_files.append(file)
                seen.add(key)
    
    print("=" * 80)
    print("BATCH TIFF STRIP RECOVERY")
    print("=" * 80)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(tiff_files)} TIFF files")
    print("=" * 80)
    
    results = {
        'good_files': [],
        'full_recovery': [],
        'partial_recovery': [],
        'no_recovery': []
    }
    
    for i, filepath in enumerate(sorted(tiff_files), 1):
        print(f"\n[{i}/{len(tiff_files)}]")
        
        # Check if this file is in the known-good list
        if filepath.name.lower() in good_files_set:
            print(f"Processing: {filepath.name}")
            print("=" * 70)
            print("✅ File is in good_files.txt - copying directly...")
            
            try:
                # Just copy the good file
                output_filename = f"good_{filepath.stem}.png"
                output_path_file = output_path / output_filename
                
                # Convert TIFF to PNG using PIL (should work for known-good files)
                with Image.open(filepath) as img:
                    img.save(output_path_file, 'PNG')
                    
                    # Create thumbnail
                    img.thumbnail((320, 240), Image.Resampling.LANCZOS)
                    thumb_path = output_path / f"thumb_{output_filename}"
                    img.save(thumb_path, 'JPEG', quality=90)
                
                print(f"   Saved as: {output_filename}")
                
                results['good_files'].append({
                    'filename': filepath.name,
                    'output': output_filename,
                    'status': 'good'
                })
            except Exception as e:
                print(f"   ⚠️  Could not convert: {e}")
                print(f"   Trying strip recovery instead...")
                result = recover_tiff_by_strips(str(filepath), str(output_path))
                if result:
                    if result.get('status') == 'good':
                        results['good_files'].append(result)
                    elif result['strips_total'] > 0:
                        recovery_percent = (result['strips_recovered'] / result['strips_total']) * 100
                        if recovery_percent >= 90:
                            results['full_recovery'].append(result)
                        else:
                            results['partial_recovery'].append(result)
                else:
                    results['no_recovery'].append(filepath.name)
            continue
        
        result = recover_tiff_by_strips(str(filepath), str(output_path))
        
        if result:
            if result.get('status') == 'good':
                results['good_files'].append(result)
            elif result['strips_total'] > 0:
                recovery_percent = (result['strips_recovered'] / result['strips_total']) * 100
                if recovery_percent >= 90:
                    results['full_recovery'].append(result)
                else:
                    results['partial_recovery'].append(result)
        else:
            results['no_recovery'].append(filepath.name)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total files processed: {len(tiff_files)}")
    print(f"✅ Good files (no recovery needed): {len(results['good_files'])}")
    print(f"✅ Full recovery (>90%): {len(results['full_recovery'])}")
    print(f"⚠️  Partial recovery: {len(results['partial_recovery'])}")
    print(f"❌ No recovery: {len(results['no_recovery'])}")
    print("=" * 80)
    
    # Detailed stats for partial recoveries
    if results['partial_recovery']:
        print("\nPARTIAL RECOVERIES:")
        print("-" * 80)
        for r in results['partial_recovery']:
            recovery_rate = (r['strips_recovered'] / r['strips_total']) * 100
            row_recovery = (r['rows_recovered'] / r['rows_total']) * 100
            print(f"  {r['filename']:<50} {recovery_rate:5.1f}% ({r['rows_recovered']}/{r['rows_total']} rows)")
    
    # Save detailed JSON report
    report_path = output_path / "batch_recovery_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'processed_date': datetime.now().isoformat(),
            'source_directory': str(source_dir),
            'total_files': len(tiff_files),
            'good_files': results['good_files'],
            'full_recovery': results['full_recovery'],
            'partial_recovery': results['partial_recovery'],
            'no_recovery': results['no_recovery']
        }, f, indent=2)
    
    print(f"\nDetailed report saved: {report_path}")
    
    # Save simple summary
    summary_path = output_path / "RECOVERY_SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write(f"TIFF Strip Recovery Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Source: {source_dir}\n")
        f.write(f"Total files: {len(tiff_files)}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Good files (no recovery needed): {len(results['good_files'])}\n")
        f.write(f"  Full recovery (>90%): {len(results['full_recovery'])}\n")
        f.write(f"  Partial recovery: {len(results['partial_recovery'])}\n")
        f.write(f"  No recovery: {len(results['no_recovery'])}\n\n")
        
        if results['good_files']:
            f.write(f"Good Files:\n")
            f.write(f"-" * 80 + "\n")
            for r in results['good_files']:
                f.write(f"  {r['filename']}\n")
            f.write(f"\n")
        
        if results['partial_recovery']:
            f.write(f"Partial Recovery Details:\n")
            f.write(f"-" * 80 + "\n")
            for r in results['partial_recovery']:
                recovery_rate = (r['strips_recovered'] / r['strips_total']) * 100
                f.write(f"  {r['filename']}: {recovery_rate:.1f}% recovered\n")
    
    print(f"Summary saved: {summary_path}")
    print("\n" + "=" * 80)
    print(f"Recovered images are in: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_strip_recovery.py <source_directory> [output_directory]")
        print()
        print("Example:")
        print('  python batch_strip_recovery.py "D:\\broken" "D:\\broken\\strip_recovered"')
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(source_dir, "strip_recovered")
    
    batch_process_directory(source_dir, output_dir)
