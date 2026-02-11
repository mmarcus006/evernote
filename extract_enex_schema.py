#!/usr/bin/env python3
"""
Extract the XML schema from Evernote .enex export files.

Parses all .enex files in the enex_files/ directory and produces a detailed
schema report showing:
  - Element hierarchy (parent/child nesting)
  - Attributes with occurrence counts and sample values
  - Text content presence with sample values
  - Occurrence counts for every element

Also generates an XSD (XML Schema Definition) file and a JSON summary.

Usage:
    python3 extract_enex_schema.py [directory]

    directory  - path to folder containing .enex files (default: enex_files/)

Outputs:
    enex_schema.xsd   - XML Schema Definition file
    enex_schema.json  - JSON representation of the schema
    Console            - human-readable schema report
"""

import xml.etree.ElementTree as ET
import glob
import json
import os
import sys
from collections import OrderedDict


def extract_schema(enex_dir: str) -> dict:
    """Walk all .enex files and collect a schema dictionary.

    Keys are XPath-style element paths (e.g. "en-export/note/title").
    Values contain attributes, children, text info, counts, and samples.
    """
    schema: dict = {}

    def _visit(elem, parent_path=""):
        tag = elem.tag
        path = f"{parent_path}/{tag}" if parent_path else tag

        if path not in schema:
            schema[path] = {
                "attributes": {},  # attr_name -> count
                "children": set(),
                "has_text": False,
                "count": 0,
                "sample_text": None,
                "sample_attr_values": {},
                "attr_values": {},  # attr_name -> set of unique values (capped)
            }

        info = schema[path]
        info["count"] += 1

        # Collect attributes
        for attr, val in elem.attrib.items():
            info["attributes"][attr] = info["attributes"].get(attr, 0) + 1
            if attr not in info["sample_attr_values"]:
                info["sample_attr_values"][attr] = val
            if attr not in info["attr_values"]:
                info["attr_values"][attr] = set()
            if len(info["attr_values"][attr]) < 20:  # cap unique values
                info["attr_values"][attr].add(val)

        # Collect text content
        if elem.text and elem.text.strip():
            info["has_text"] = True
            txt = elem.text.strip()
            if info["sample_text"] is None and len(txt) < 200:
                info["sample_text"] = txt

        # Recurse into children
        for child in elem:
            info["children"].add(child.tag)
            _visit(child, path)

    enex_files = sorted(glob.glob(os.path.join(enex_dir, "*.enex")))
    if not enex_files:
        print(f"No .enex files found in {enex_dir!r}", file=sys.stderr)
        sys.exit(1)

    for filepath in enex_files:
        filename = os.path.basename(filepath)
        print(f"  Parsing: {filename}")
        tree = ET.parse(filepath)
        _visit(tree.getroot())

    return schema


def print_schema_report(schema: dict) -> None:
    """Print a human-readable schema report to stdout."""
    total_notes = schema.get("en-export/note", {}).get("count", 0)
    total_resources = schema.get("en-export/note/resource", {}).get("count", 0)

    print(f"\n{'=' * 70}")
    print("  EVERNOTE EXPORT (.enex) XML SCHEMA REPORT")
    print(f"{'=' * 70}")
    print(f"\n  Total <note> elements:     {total_notes}")
    print(f"  Total <resource> elements: {total_resources}\n")

    for path in sorted(schema.keys()):
        info = schema[path]
        depth = path.count("/")
        indent = "  " * depth
        tag = path.split("/")[-1]

        # Print element
        print(f"{indent}<{tag}> — occurs {info['count']}x")

        # Attributes
        for attr in sorted(info["attributes"]):
            cnt = info["attributes"][attr]
            sample = info["sample_attr_values"].get(attr, "")
            unique_vals = info["attr_values"].get(attr, set())
            if len(unique_vals) <= 5:
                vals_str = ", ".join(repr(v) for v in sorted(unique_vals))
                print(f"{indent}  @{attr}  ({cnt}x)  values: [{vals_str}]")
            else:
                print(
                    f"{indent}  @{attr}  ({cnt}x)  sample: {sample!r}"
                    f"  ({len(unique_vals)}+ unique values)"
                )

        # Children
        if info["children"]:
            print(f"{indent}  children: {', '.join(sorted(info['children']))}")

        # Text content
        if info["has_text"]:
            sample = info["sample_text"] or "(content too long to sample)"
            print(f"{indent}  text content  sample: {sample!r}")

        print()


def generate_xsd(schema: dict, output_path: str) -> None:
    """Generate an XSD file representing the discovered schema."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">',
        "",
        "  <!-- Auto-generated XSD from Evernote .enex export files -->",
        "  <!-- This schema was inferred from actual data, not the official DTD -->",
        "",
    ]

    # Build a lookup: for each element path, we know its children, attrs, text
    # We need to define complex types for elements with children/attrs,
    # and simple types for leaf text elements.

    # Gather all unique element definitions by their path
    # We'll create named types for each path

    def _type_name(path: str) -> str:
        """Convert a path like en-export/note/title to EnExportNoteTitleType."""
        parts = path.split("/")
        return (
            "".join(p.replace("-", " ").title().replace(" ", "") for p in parts)
            + "Type"
        )

    def _elem_name(path: str) -> str:
        return path.split("/")[-1]

    # Determine which children are optional (count < parent count)
    parent_counts = {}
    for path, info in schema.items():
        parent_counts[path] = info["count"]

    # Root element
    root_path = "en-export"

    lines.append(f'  <xs:element name="en-export" type="{_type_name(root_path)}"/>')
    lines.append("")

    for path in sorted(schema.keys()):
        info = schema[path]
        tname = _type_name(path)
        has_children = bool(info["children"])
        has_attrs = bool(info["attributes"])
        has_text = info["has_text"]

        if has_children or has_attrs:
            if has_text and not has_children and has_attrs:
                # Element with text content and attributes -> simpleContent extension
                lines.append(f'  <xs:complexType name="{tname}">')
                lines.append("    <xs:simpleContent>")
                lines.append('      <xs:extension base="xs:string">')
                for attr in sorted(info["attributes"]):
                    required = info["attributes"][attr] >= info["count"]
                    use = "required" if required else "optional"
                    lines.append(
                        f'        <xs:attribute name="{attr}"'
                        f' type="xs:string" use="{use}"/>'
                    )
                lines.append("      </xs:extension>")
                lines.append("    </xs:simpleContent>")
                lines.append("  </xs:complexType>")
            else:
                if has_text and has_children:
                    lines.append(f'  <xs:complexType name="{tname}" mixed="true">')
                else:
                    lines.append(f'  <xs:complexType name="{tname}">')

            if has_children:
                lines.append("    <xs:sequence>")
                for child_tag in sorted(info["children"]):
                    child_path = f"{path}/{child_tag}"
                    child_info = schema.get(child_path, {})
                    child_count = child_info.get("count", 0)
                    parent_count = info["count"]

                    min_occurs = "1" if child_count >= parent_count else "0"
                    # Check if any note has more than one of this child
                    max_occurs = "unbounded" if child_count > parent_count else "1"

                    child_has_children = bool(child_info.get("children"))
                    child_has_attrs = bool(child_info.get("attributes"))

                    if child_has_children or child_has_attrs:
                        type_ref = f' type="{_type_name(child_path)}"'
                    elif child_info.get("has_text"):
                        type_ref = ' type="xs:string"'
                    else:
                        type_ref = ' type="xs:string"'

                    lines.append(
                        f'      <xs:element name="{child_tag}"{type_ref}'
                        f' minOccurs="{min_occurs}" maxOccurs="{max_occurs}"/>'
                    )
                lines.append("    </xs:sequence>")

            for attr in sorted(info["attributes"]):
                required = info["attributes"][attr] >= info["count"]
                use = "required" if required else "optional"
                lines.append(
                    f'    <xs:attribute name="{attr}" type="xs:string" use="{use}"/>'
                )

            lines.append("  </xs:complexType>")
        else:
            # Simple text-only element — no need for a named type,
            # referenced as xs:string inline
            pass

        lines.append("")

    lines.append("</xs:schema>")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  XSD written to: {output_path}")


def generate_json(schema: dict, output_path: str) -> None:
    """Generate a JSON representation of the schema."""
    json_schema = OrderedDict()

    for path in sorted(schema.keys()):
        info = schema[path]
        json_schema[path] = OrderedDict(
            [
                ("element", path.split("/")[-1]),
                ("occurrences", info["count"]),
                (
                    "attributes",
                    {
                        attr: {
                            "count": info["attributes"][attr],
                            "sample": info["sample_attr_values"].get(attr, ""),
                            "unique_values": sorted(
                                info["attr_values"].get(attr, set())
                            )
                            if len(info["attr_values"].get(attr, set())) <= 10
                            else f"{len(info['attr_values'].get(attr, set()))}+ unique",
                        }
                        for attr in sorted(info["attributes"])
                    },
                ),
                ("children", sorted(info["children"])),
                ("has_text_content", info["has_text"]),
                ("sample_text", info["sample_text"]),
            ]
        )

    with open(output_path, "w") as f:
        json.dump(json_schema, f, indent=2, default=str)
    print(f"  JSON written to: {output_path}")


def main():
    enex_dir = sys.argv[1] if len(sys.argv) > 1 else "enex_files"

    print(f"\nExtracting XML schema from .enex files in: {enex_dir}/\n")
    schema = extract_schema(enex_dir)

    # Console report
    print_schema_report(schema)

    # XSD output
    xsd_path = os.path.join(os.path.dirname(enex_dir.rstrip("/")), "enex_schema.xsd")
    generate_xsd(schema, xsd_path)

    # JSON output
    json_path = os.path.join(os.path.dirname(enex_dir.rstrip("/")), "enex_schema.json")
    generate_json(schema, json_path)

    total_enex_files = len(glob.glob(os.path.join(enex_dir, "*.enex")))
    print(f"\nDone. Processed {total_enex_files} file(s).\n")


if __name__ == "__main__":
    main()
