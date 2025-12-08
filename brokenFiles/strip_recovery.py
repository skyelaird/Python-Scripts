#!/usr/bin/env python3
"""
Strip-by-Strip TIFF Recovery - FIXED VERSION
Attempts to extract individual strips from corrupted TIFFs
Now skips files that open normally and handles multiple compression types
"""

import os
import tifffile
import numpy as np
from PIL import Image
import zlib

def test_if_file_is_good(filepath):
    """Test if file opens normally - try both tifffile and PIL"""
    # First try tifffile (more forgiving)
    try:
        with tifffile.TiffFile(filepath) as tif:
            page = tif.pages[0]
            # Try to read the actual image data
            data = page.asarray()
            if data is not None and data.size > 0:
                return True
    except:
        pass
    
    # Fall back to PIL
    try:
        with Image.open(filepath) as img:
            img.load()  # Force full load
            return True
    except:
        pass
    
    return False

def recover_tiff_by_strips(filepath, output_dir):
    """
    Attempt to recover a TIFF by processing each strip individually.
    Create a partial image from whatever strips succeed.
    """
    
    filename = os.path.basename(filepath)
    print(f"\n{'='*70}")
    print(f"Processing: {filename}")
    print(f"{'='*70}")
    
    # First, check if file is actually good
    if test_if_file_is_good(filepath):
        print("✅ File opens normally with PIL - no recovery needed!")
        print("   This is a GOOD file, copying to output...")
        
        # Copy to output as-is
        try:
            with Image.open(filepath) as img:
                output_filename = f"good_{filename.replace('.tif', '.png').replace('.TIF', '.png')}"
                output_path = os.path.join(output_dir, output_filename)
                img.save(output_path, 'PNG')
                
                # Create thumbnail
                img.thumbnail((320, 240), Image.Resampling.LANCZOS)
                thumb_path = os.path.join(output_dir, f"thumb_{output_filename}")
                img.save(thumb_path, 'JPEG', quality=90)
                
                print(f"   Saved as: {output_filename}")
                
                return {
                    'filename': filename,
                    'status': 'good',
                    'strips_total': 0,
                    'strips_recovered': 0,
                    'rows_recovered': 0,
                    'rows_total': 0,
                    'output': output_filename
                }
        except Exception as e:
            print(f"   Warning: Could not copy - {e}")
            return None
    
    try:
        with tifffile.TiffFile(filepath) as tif:
            page = tif.pages[0]
            
            print(f"Image info:")
            print(f"  Dimensions: {page.shape}")
            print(f"  Compression: {page.compression}")
            print(f"  Strips: {len(page.dataoffsets) if hasattr(page, 'dataoffsets') else 'unknown'}")
            
            height, width = page.shape[:2]
            channels = page.shape[2] if len(page.shape) > 2 else 1
            
            # Create empty array for recovered image
            if channels > 1:
                recovered = np.zeros((height, width, channels), dtype=np.uint8)
            else:
                recovered = np.zeros((height, width), dtype=np.uint8)
            
            # Track which rows we successfully recovered
            recovered_rows = set()
            
            # Try to get strip information
            if not hasattr(page, 'dataoffsets'):
                print("  ❌ No strip information available")
                return None
            
            strip_offsets = page.dataoffsets
            strip_byte_counts = page.databytecounts
            rows_per_strip = page.tags.get('RowsPerStrip', page.tags.get(278))
            
            if rows_per_strip:
                rows_per_strip = rows_per_strip.value
            else:
                # Estimate
                rows_per_strip = height // len(strip_offsets)
            
            print(f"  Rows per strip: {rows_per_strip}")
            print(f"\nAttempting strip recovery...")
            
            with open(filepath, 'rb') as f:
                successful_strips = 0
                
                for strip_num in range(len(strip_offsets)):
                    strip_start_row = strip_num * rows_per_strip
                    strip_end_row = min(strip_start_row + rows_per_strip, height)
                    
                    try:
                        # Read compressed strip data
                        f.seek(strip_offsets[strip_num])
                        compressed_data = f.read(strip_byte_counts[strip_num])
                        
                        decompressed = None
                        
                        # Try different compression types
                        if page.compression == 1:  # No compression
                            decompressed = compressed_data
                        elif page.compression == 5:  # LZW
                            print(f"  Strip {strip_num:2d}: ⚠️  LZW compression not supported")
                            continue
                        elif page.compression == 8:  # Deflate
                            try:
                                decompressed = zlib.decompress(compressed_data)
                            except zlib.error:
                                print(f"  Strip {strip_num:2d}: ✗ Decompress failed")
                                continue
                        else:
                            print(f"  Strip {strip_num:2d}: ⚠️  Unsupported compression: {page.compression}")
                            continue
                        
                        if decompressed:
                            # Convert bytes to array
                            expected_size = (strip_end_row - strip_start_row) * width * channels
                            
                            if len(decompressed) >= expected_size:
                                strip_array = np.frombuffer(decompressed[:expected_size], dtype=np.uint8)
                                strip_array = strip_array.reshape((strip_end_row - strip_start_row, width, channels) if channels > 1 else (strip_end_row - strip_start_row, width))
                                
                                # Place in recovered image
                                if channels > 1:
                                    recovered[strip_start_row:strip_end_row, :, :] = strip_array
                                else:
                                    recovered[strip_start_row:strip_end_row, :] = strip_array
                                
                                successful_strips += 1
                                for row in range(strip_start_row, strip_end_row):
                                    recovered_rows.add(row)
                                
                                print(f"  Strip {strip_num:2d}: ✓ Rows {strip_start_row:4d}-{strip_end_row:4d}")
                                
                    except Exception as e:
                        print(f"  Strip {strip_num:2d}: ✗ {type(e).__name__}")
                
                print(f"\n{'='*70}")
                print(f"Recovery Summary:")
                print(f"  Total strips: {len(strip_offsets)}")
                print(f"  Successful: {successful_strips}")
                print(f"  Failed: {len(strip_offsets) - successful_strips}")
                print(f"  Recovery rate: {successful_strips/len(strip_offsets)*100:.1f}%")
                print(f"  Rows recovered: {len(recovered_rows)}/{height} ({len(recovered_rows)/height*100:.1f}%)")
                
                if successful_strips > 0:
                    # Save recovered image
                    img = Image.fromarray(recovered)
                    
                    output_filename = f"recovered_{filename.replace('.tif', '.png').replace('.TIF', '.png')}"
                    output_path = os.path.join(output_dir, output_filename)
                    img.save(output_path, 'PNG')
                    
                    print(f"\n✅ Partial image saved: {output_filename}")
                    print(f"   {len(recovered_rows)} rows recovered out of {height}")
                    
                    # Also create a thumbnail
                    img.thumbnail((320, 240), Image.Resampling.LANCZOS)
                    thumb_path = os.path.join(output_dir, f"thumb_{output_filename}")
                    img.save(thumb_path, 'JPEG', quality=90)
                    print(f"   Thumbnail: thumb_{output_filename}")
                    
                    return {
                        'filename': filename,
                        'status': 'partial',
                        'strips_total': len(strip_offsets),
                        'strips_recovered': successful_strips,
                        'rows_recovered': len(recovered_rows),
                        'rows_total': height,
                        'output': output_filename
                    }
                else:
                    print(f"\n❌ No strips could be recovered")
                    return None
                    
    except Exception as e:
        print(f"❌ Error opening file: {type(e).__name__}: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python strip_recovery.py <tiff_file> [output_dir]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./recovered_strips"
    
    os.makedirs(output_dir, exist_ok=True)
    
    result = recover_tiff_by_strips(filepath, output_dir)
    
    if result:
        print(f"\n{'='*70}")
        print("SUCCESS!")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("No recovery possible")
        print(f"{'='*70}")
