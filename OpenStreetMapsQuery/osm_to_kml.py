#!/usr/bin/env python3
"""
OSM Airport to KML Converter
Queries OpenStreetMap Overpass API for airport aeroway data and generates KML

Usage: python osm_to_kml.py CYHZ
"""

import sys
import json
import requests
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def query_overpass(icao_code):
    """
    Query Overpass API for airport aeroway data
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Overpass QL query for airport features
    query = f"""
    [out:json][timeout:25];
    (
      // Find airport by ICAO code
      relation["aeroway"="aerodrome"]["icao"="{icao_code}"];
      way["aeroway"="aerodrome"]["icao"="{icao_code}"];
      node["aeroway"="aerodrome"]["icao"="{icao_code}"];
    )->.airport;
    
    // Get all aeroway features within or near the airport
    (
      // Runways
      way["aeroway"="runway"](around.airport:2000);
      // Taxiways
      way["aeroway"="taxiway"](around.airport:2000);
      // Aprons
      way["aeroway"="apron"](around.airport:2000);
      // Parking positions
      node["aeroway"="parking_position"](around.airport:2000);
      // Gates
      node["aeroway"="gate"](around.airport:2000);
      // Holding positions
      node["aeroway"="holding_position"](around.airport:2000);
    );
    
    out geom;
    """
    
    print(f"Querying Overpass API for {icao_code}...")
    response = requests.post(overpass_url, data={'data': query}, timeout=30)
    
    if response.status_code != 200:
        raise Exception(f"Overpass API error: {response.status_code}")
    
    return response.json()

def create_kml_document(icao_code):
    """
    Create base KML document structure
    """
    kml = Element('kml', xmlns="http://www.opengis.net/kml/2.2")
    document = SubElement(kml, 'Document')
    
    name = SubElement(document, 'name')
    name.text = f"{icao_code} Ground Network"
    
    # Define styles for different aeroway types
    styles = {
        'runway': {'color': 'ff0000ff', 'width': '4'},      # Red
        'taxiway': {'color': 'ff00ffff', 'width': '2'},     # Yellow
        'apron': {'color': 'ff808080', 'width': '1'},       # Gray
        'parking': {'color': 'ff00ff00', 'scale': '0.5'},   # Green
        'gate': {'color': 'ff0000ff', 'scale': '0.5'},      # Blue
        'holding': {'color': 'ffff00ff', 'scale': '0.4'},   # Magenta
    }
    
    for style_id, style_attrs in styles.items():
        style = SubElement(document, 'Style', id=style_id)
        if 'color' in style_attrs and 'width' in style_attrs:
            linestyle = SubElement(style, 'LineStyle')
            color = SubElement(linestyle, 'color')
            color.text = style_attrs['color']
            width = SubElement(linestyle, 'width')
            width.text = style_attrs['width']
        if 'scale' in style_attrs:
            iconstyle = SubElement(style, 'IconStyle')
            scale = SubElement(iconstyle, 'scale')
            scale.text = style_attrs['scale']
            color = SubElement(iconstyle, 'color')
            color.text = style_attrs['color']
    
    return kml, document

def add_way_to_kml(document, element, style_id):
    """
    Add an OSM way (line) to KML
    """
    placemark = SubElement(document, 'Placemark')
    
    # Add name if available
    name_tag = element.get('tags', {}).get('name') or element.get('tags', {}).get('ref', '')
    if name_tag:
        name = SubElement(placemark, 'name')
        name.text = name_tag
    
    # Add description with tags
    desc_parts = []
    tags = element.get('tags', {})
    for key, value in tags.items():
        if key not in ['name', 'ref', 'icao', 'aeroway']:
            desc_parts.append(f"{key}: {value}")
    
    if desc_parts:
        description = SubElement(placemark, 'description')
        description.text = '\n'.join(desc_parts)
    
    # Add style
    styleurl = SubElement(placemark, 'styleUrl')
    styleurl.text = f"#{style_id}"
    
    # Add geometry
    geometry = element.get('geometry', [])
    if not geometry:
        return
    
    # Determine if it's a closed polygon (apron) or line (runway/taxiway)
    is_closed = (geometry[0]['lat'] == geometry[-1]['lat'] and 
                 geometry[0]['lon'] == geometry[-1]['lon'])
    
    if is_closed and style_id == 'apron':
        polygon = SubElement(placemark, 'Polygon')
        outer = SubElement(polygon, 'outerBoundaryIs')
        linearring = SubElement(outer, 'LinearRing')
        coordinates = SubElement(linearring, 'coordinates')
    else:
        linestring = SubElement(placemark, 'LineString')
        coordinates = SubElement(linestring, 'coordinates')
    
    # Add coordinates (KML format: lon,lat,alt)
    coord_text = []
    for node in geometry:
        coord_text.append(f"{node['lon']},{node['lat']},0")
    coordinates.text = ' '.join(coord_text)

def add_node_to_kml(document, element, style_id):
    """
    Add an OSM node (point) to KML
    """
    placemark = SubElement(document, 'Placemark')
    
    # Add name if available
    name_tag = element.get('tags', {}).get('name') or element.get('tags', {}).get('ref', '')
    if name_tag:
        name = SubElement(placemark, 'name')
        name.text = name_tag
    
    # Add description with tags
    desc_parts = []
    tags = element.get('tags', {})
    for key, value in tags.items():
        if key not in ['name', 'ref', 'icao', 'aeroway']:
            desc_parts.append(f"{key}: {value}")
    
    if desc_parts:
        description = SubElement(placemark, 'description')
        description.text = '\n'.join(desc_parts)
    
    # Add style
    styleurl = SubElement(placemark, 'styleUrl')
    styleurl.text = f"#{style_id}"
    
    # Add point geometry
    point = SubElement(placemark, 'Point')
    coordinates = SubElement(point, 'coordinates')
    coordinates.text = f"{element['lon']},{element['lat']},0"

def convert_osm_to_kml(osm_data, icao_code):
    """
    Convert OSM data to KML
    """
    kml, document = create_kml_document(icao_code)
    
    # Create folders for organization
    runway_folder = SubElement(document, 'Folder')
    SubElement(runway_folder, 'name').text = 'Runways'
    
    taxiway_folder = SubElement(document, 'Folder')
    SubElement(taxiway_folder, 'name').text = 'Taxiways'
    
    apron_folder = SubElement(document, 'Folder')
    SubElement(apron_folder, 'name').text = 'Aprons'
    
    parking_folder = SubElement(document, 'Folder')
    SubElement(parking_folder, 'name').text = 'Parking & Gates'
    
    # Process elements
    for element in osm_data.get('elements', []):
        aeroway_type = element.get('tags', {}).get('aeroway')
        
        if element['type'] == 'way':
            if aeroway_type == 'runway':
                add_way_to_kml(runway_folder, element, 'runway')
            elif aeroway_type == 'taxiway':
                add_way_to_kml(taxiway_folder, element, 'taxiway')
            elif aeroway_type == 'apron':
                add_way_to_kml(apron_folder, element, 'apron')
        
        elif element['type'] == 'node':
            if aeroway_type == 'parking_position':
                add_node_to_kml(parking_folder, element, 'parking')
            elif aeroway_type == 'gate':
                add_node_to_kml(parking_folder, element, 'gate')
            elif aeroway_type == 'holding_position':
                add_node_to_kml(parking_folder, element, 'holding')
    
    return kml

def prettify_xml(elem):
    """
    Return a pretty-printed XML string
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def main():
    if len(sys.argv) != 2:
        print("Usage: python osm_to_kml.py ICAO_CODE")
        print("Example: python osm_to_kml.py CYHZ")
        sys.exit(1)
    
    icao_code = sys.argv[1].upper()
    output_file = f"{icao_code}_ground.kml"
    
    try:
        # Query OSM
        osm_data = query_overpass(icao_code)
        
        if not osm_data.get('elements'):
            print(f"No aeroway data found for {icao_code}")
            print("This could mean:")
            print("  - The airport is not in OpenStreetMap")
            print("  - The ICAO code is incorrect")
            print("  - The airport lacks detailed ground network data in OSM")
            sys.exit(1)
        
        print(f"Found {len(osm_data['elements'])} aeroway features")
        
        # Convert to KML
        kml = convert_osm_to_kml(osm_data, icao_code)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prettify_xml(kml))
        
        print(f"Successfully created {output_file}")
        print(f"\nYou can now:")
        print(f"  1. Open {output_file} in Google Earth to verify")
        print(f"  2. Use it as input to your mapbuilder tool")
        
    except requests.exceptions.RequestException as e:
        print(f"Error querying Overpass API: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
