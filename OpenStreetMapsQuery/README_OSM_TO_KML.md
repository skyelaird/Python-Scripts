# OSM Airport Ground Network to KML Converter

Extracts airport ground network data from OpenStreetMap and converts it to KML format for use with EuroScope/TopSky mapbuilder.

## Requirements

```bash
pip install requests
```

## Usage

```bash
python osm_to_kml.py ICAO_CODE
```

Examples:
```bash
python osm_to_kml.py CYHZ  # Halifax Stanfield
python osm_to_kml.py CYUL  # Montreal Trudeau
python osm_to_kml.py CYYZ  # Toronto Pearson
python osm_to_kml.py CYVR  # Vancouver International
```

## Output

Creates `{ICAO}_ground.kml` containing:

- **Runways** - Red lines (4px width)
- **Taxiways** - Yellow lines (2px width) with designators (A, B, C, etc.)
- **Aprons** - Gray polygons
- **Parking Positions** - Green points
- **Gates** - Blue points
- **Holding Positions** - Magenta points

## Workflow

1. **Generate KML**: `python osm_to_kml.py CYHZ`
2. **Verify**: Open `CYHZ_ground.kml` in Google Earth
3. **Process**: Feed KML into your mapbuilder tool
4. **Output**: Generate EuroScope .sct/.ese files

## Data Source

OpenStreetMap (ODbL license) via Overpass API

Quality depends on OSM coverage:
- Major Canadian airports (CYYZ, CYUL, CYVR, CYHZ) - Usually excellent
- Regional airports - Coverage varies
- Small airports - May have limited or no data

## OpenStreetMap Data Quality

Check OSM coverage before running:
1. Visit https://www.openstreetmap.org
2. Search for your airport ICAO code
3. Look for purple aeroway features (runways, taxiways)
4. If missing/incomplete, you can contribute to OSM!

## Limitations

- Only extracts what's in OpenStreetMap
- No official NAV CANADA data (proprietary)
- Coordinates are WGS84 (may need conversion for some tools)
- Taxiway widths not included (would need manual adjustment)

## Alternative Data Sources

If OSM data is insufficient:
1. **Manual tracing**: Use Google Earth Pro to trace taxiways
2. **Airport diagrams**: Georeferenced PDFs from NAV CANADA AIP
3. **Community sources**: VATSIM/VATCAN community-created files

## Legal

- OSM data: ODbL license (Open Database License)
- Safe for VATSIM use
- No NAV CANADA copyright concerns
- Attribution: "Data © OpenStreetMap contributors"

## Integration with mapbuilder

Your mapbuilder tool should accept KML as input. The generated KML includes:
- Standard KML 2.2 format
- LineStrings for linear features (runways, taxiways)
- Polygons for areas (aprons)
- Points for positions (gates, parking)
- Names and metadata in tags

## Troubleshooting

**"No aeroway data found"**
- Airport not in OSM, or ICAO code incorrect
- Check https://www.openstreetmap.org for coverage

**Network errors**
- Overpass API may be rate-limited
- Try again in a few minutes
- Use alternative: https://overpass.kumi.systems/api/interpreter

**Incomplete data**
- OSM data quality varies by airport
- Consider contributing missing data to OSM
- Use hybrid approach: OSM + manual adjustments

## For CZQM/CZQX vACC

Major airports in your FIRs with good OSM coverage:
- CYHZ (Halifax) - Excellent
- CYYT (St. John's) - Good
- CYYG (Charlottetown) - Good
- CYSJ (Saint John) - Good
- CYQM (Moncton) - Good
- CYQX (Gander) - Good

Smaller airports may need manual work or community sourcing.

## Contact

Issues or improvements: Open issue on skyelaird/CZQM-vACC or mapbuilder repo
