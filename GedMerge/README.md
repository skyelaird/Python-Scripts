# GEDMerge

A powerful tool to find and merge duplicate people in GEDCOM genealogy files.

## Overview

GEDMerge is designed to help genealogists clean up their family tree data by automatically detecting and merging duplicate individual records in GEDCOM files. The tool uses advanced matching algorithms including fuzzy string matching and phonetic comparison to identify potential duplicates with high accuracy.

## Features

- **GEDCOM Parsing**: Full support for GEDCOM 5.5 and 5.5.1 formats
- **Duplicate Detection**: Advanced algorithms to find potential duplicate individuals
- **Smart Merging**: Intelligent merging that preserves all relevant data
- **Data Integrity**: Maintains relationships between individuals and families
- **CLI Interface**: Easy-to-use command-line interface

## Project Structure

```
GedMerge/
├── gedmerge/
│   ├── core/           # Core data models and GEDCOM parsing
│   ├── matching/       # Duplicate detection algorithms
│   ├── merge/          # Merge logic and strategies
│   ├── ui/             # User interface (CLI)
│   ├── utils/          # Utility functions
│   └── data/           # Data storage and caching
├── tests/              # Test suite
├── GEDCOM/             # Sample GEDCOM files
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # This file
```

## Installation

### From Source

```bash
cd GedMerge
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Dependencies

- **python-gedcom**: GEDCOM file parsing
- **rapidfuzz**: Fast fuzzy string matching
- **phonetics**: Phonetic matching algorithms (Soundex, Metaphone, etc.)
- **pytest**: Testing framework (dev dependency)

## Quick Start

### Load and Analyze a GEDCOM File

```bash
gedmerge analyze path/to/your/file.ged
```

This will display basic statistics about your GEDCOM file including:
- Number of individuals
- Number of families
- Date range
- Geographic coverage

### Find Duplicates

```bash
gedmerge find-duplicates path/to/your/file.ged
```

### Merge Duplicates

```bash
gedmerge merge path/to/your/file.ged --output merged.ged
```

## Development

### Running Tests

```bash
pytest
```

### Running Tests with Coverage

```bash
pytest --cov=gedmerge --cov-report=html
```

## Roadmap

### Phase 1: Foundation (Current)
- [x] Project setup and structure
- [x] GEDCOM parser
- [x] Core data models
- [x] Basic CLI
- [ ] Unit tests

### Phase 2: Duplicate Detection
- [ ] Name matching algorithms
- [ ] Date/place comparison
- [ ] Relationship analysis
- [ ] Scoring system

### Phase 3: Merging Logic
- [ ] Merge strategies
- [ ] Conflict resolution
- [ ] Data preservation
- [ ] Undo capability

### Phase 4: User Interface
- [ ] Interactive CLI
- [ ] Review and approve duplicates
- [ ] Batch operations
- [ ] Progress tracking

### Phase 5: Advanced Features
- [ ] Machine learning for matching
- [ ] Custom matching rules
- [ ] Performance optimization
- [ ] Export reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- GEDCOM format specification by The Church of Jesus Christ of Latter-day Saints
- python-gedcom library contributors
- Genealogy community for feature suggestions and testing

## Support

For issues, questions, or suggestions, please open an issue on the project repository.
