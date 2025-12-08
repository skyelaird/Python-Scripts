# TIFF Recovery Toolkit

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run batch recovery on all files:**
```bash
python batch_strip_recovery.py "D:\broken"
```

This will process all 80 corrupted TIFFs and save recovered images to `D:\broken\strip_recovered\`

## What It Does

- Extracts working strips from corrupted TIFFs
- Creates partial images from whatever data is recoverable
- Generates thumbnails of recovered content
- Produces detailed recovery report

## Proven Results

Test file showed **8.6% recovery** (117/1365 rows) from a "completely corrupted" file.

**This proves the technique works!** Early strips survive Drobo crashes.

## Expected Recovery Rate

- **10-30 files**: Likely to have partial recoverable data (5-50%)
- **Pattern**: Top portions of images (where write started)
- **Useful**: Even partial recovery might capture faces/subjects

## Files in This Toolkit

- `batch_strip_recovery.py` - Process all TIFFs at once (**USE THIS**)
- `strip_recovery.py` - Single file recovery (auto-loaded by batch script)
- `requirements.txt` - Python dependencies
- `Validate-Images.ps1` - Initial validation script (already run)

## After Running

Check these output files:
- `D:\broken\strip_recovered\RECOVERY_SUMMARY.txt` - Human-readable stats
- `D:\broken\strip_recovered\batch_recovery_report.json` - Detailed data
- `D:\broken\strip_recovered\recovered_*.png` - Recovered images
- `D:\broken\strip_recovered\thumb_*.png` - Thumbnails

## Next Steps

After batch processing completes, review the summary to see:
- How many files had full recovery (>90%)
- How many had partial recovery (useful data)
- Which files are worth keeping vs. deleting

---

**Ready to recover your images?**

```bash
python batch_strip_recovery.py "D:\broken"
```

Let it run and check the results!

---

Joel Morin - VE1ATM - December 2025
