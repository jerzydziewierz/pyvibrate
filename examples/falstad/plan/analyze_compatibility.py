#!/usr/bin/env python3
"""
Analyze Falstad circuit files for compatibility with pyvibrate.

This script scans all Falstad circuit .txt files and determines:
1. Which circuits are compatible (use only supported elements)
2. Which circuits have Python implementations
3. Which circuits are incompatible and why (first blocking element)

Usage:
    python analyze_compatibility.py [--falstad-dir PATH] [--output FORMAT]

Output formats: table, markdown, csv
"""

import os
import sys
import argparse
from pathlib import Path


# Default paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent  # pyvibrate root
FALSTAD_DIR_DEFAULT = PROJECT_DIR.parent / "from-source" / "circuit-simulator" / "src" / "circuits"
EXAMPLES_DIR = SCRIPT_DIR.parent  # examples/falstad/


# Our Python implementations: falstad_file -> our_python_path (relative to examples/falstad/)
OUR_IMPLEMENTATIONS = {
    "voltdivide.txt": "basic/voltage_divider.py",
    "filt-lopass.txt": "filters/lowpass_rc.py",
    "lrc.txt": "rlc/lrc_resonance.py",
    "inductkick.txt": "rlc/inductor_kick.py",
    "amp-invert.txt": "opamp/inverting.py",
}


# Supported element codes in pyvibrate
SUPPORTED_ELEMENTS = {
    'r',    # Resistor
    'c',    # Capacitor
    'l',    # Inductor
    'v',    # Voltage source (DC)
    'R',    # Voltage source (with waveform options)
    'a',    # Op-amp (ideal)
    's',    # Switch SPST
    'S',    # Switch SPDT
    'w',    # Wire
    'g',    # Ground
    '$',    # Simulation parameters (metadata)
    'o',    # Oscilloscope probe (display only)
    'O',    # Output probe (display only)
    'h',    # Hint text (display only)
    'x',    # Text annotation (display only)
    '170',  # AC voltage source (sine)
    '174',  # VCVS (Voltage-Controlled Voltage Source)
}


# Element code to human-readable name mapping
ELEMENT_NAMES = {
    # Semiconductors
    't': 'Transistor (NPN)',
    'T': 'Transistor (PNP)',  # Note: also transformer in some contexts
    'f': 'MOSFET (N-ch)',
    'j': 'JFET',
    'd': 'Diode',
    'z': 'Zener diode',

    # Passive/Other
    'p': 'Potentiometer',
    'i': 'Current source',
    'M': 'Memristor',
    'A': 'Antenna',

    # Numeric codes - ICs and special elements
    '150': 'Logic inverter',
    '151': 'Logic NAND',
    '152': 'Logic NOR',
    '153': 'Logic AND',
    '154': 'Logic OR',
    '155': 'Logic XOR',
    '156': 'Logic D-FF',
    '157': 'Logic JK-FF',
    '158': 'Logic T-FF',
    '159': 'Logic counter',
    '160': 'Logic DAC',
    '161': 'Logic ADC',
    '162': 'Logic latch',
    '163': 'Logic decoder',
    '164': 'Logic mux',
    '165': '555 Timer',
    '166': 'Phase comparator',
    '167': 'VCO',
    '168': 'Relay',
    '169': 'LED',
    '171': 'Square wave src',
    '172': 'Current conveyor',
    '173': 'Comparator',
    # '174': 'VCVS',  # Supported
    '175': 'VCCS',
    '176': 'CCVS',
    '177': 'CCCS',
    '178': 'Op-amp (rail)',
    '179': 'Triode',
    '180': 'Tunnel diode',
    '181': 'SCR',
    '182': 'Triac',
    '183': 'Diac',
    '184': 'Lamp',
    '185': 'Analog switch',
    '186': 'Transmission line',
    '187': 'Varactor',
    '188': 'Photoresistor',
    '189': 'Thermistor',
    '190': 'CCII',
    '191': 'Fuse',
    '192': 'Spark gap',
    '193': 'Custom logic',
    '194': 'Custom function',
    '200': 'AM source',
    '201': 'FM source',
    '203': 'Transformer',
    '206': 'Tapped transformer',
    '207': 'Inductor (coupled)',
    '350': 'Data recorder',
    '351': 'Audio output',
    '368': 'Test point',
    '370': 'Sweep',
    '400': 'Scope XY',
    '401': 'Logic analyzer',
    '402': 'Sequence gen',
    '403': 'Audio input',
    '404': 'Stop trigger',
    '405': 'DC sweep',
    '406': 'Waveform view',
    '407': 'Ohmmeter',
    '408': 'Box',
    '409': 'Labeled node',
    '410': 'SPICE model',
}


def get_element_name(code: str) -> str:
    """Get human-readable name for element code."""
    return ELEMENT_NAMES.get(code, f'Unknown element ({code})')


def analyze_circuit(filepath: Path) -> dict:
    """
    Analyze a single Falstad circuit file.

    Returns dict with:
        - filename: str
        - compatible: bool
        - blocking_element: str or None (first incompatible element)
        - blocking_line: int or None (line number of blocking element)
        - elements_used: set of element codes used
    """
    result = {
        'filename': filepath.name,
        'compatible': True,
        'blocking_element': None,
        'blocking_element_name': None,
        'blocking_line': None,
        'elements_used': set(),
    }

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        result['compatible'] = False
        result['blocking_element'] = f'Read error: {e}'
        return result

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        code = parts[0]
        result['elements_used'].add(code)

        # Check if element is supported
        if code in SUPPORTED_ELEMENTS:
            continue

        # First incompatible element found
        if result['compatible']:
            result['compatible'] = False
            result['blocking_element'] = code
            result['blocking_element_name'] = get_element_name(code)
            result['blocking_line'] = line_num

    return result


def analyze_all_circuits(falstad_dir: Path) -> list:
    """Analyze all .txt files in the Falstad circuits directory."""
    results = []

    if not falstad_dir.exists():
        print(f"Error: Falstad directory not found: {falstad_dir}", file=sys.stderr)
        return results

    for filepath in sorted(falstad_dir.glob("*.txt")):
        result = analyze_circuit(filepath)

        # Add implementation status
        impl = OUR_IMPLEMENTATIONS.get(filepath.name, "")
        result['our_implementation'] = impl
        result['implemented'] = bool(impl)

        results.append(result)

    return results


def format_table(results: list) -> str:
    """Format results as ASCII table."""
    lines = []

    # Header
    lines.append(f"{'Falstad File':<30} {'Our Implementation':<30} {'Status':<40}")
    lines.append("-" * 100)

    # Stats
    total = len(results)
    compatible = sum(1 for r in results if r['compatible'])
    implemented = sum(1 for r in results if r['implemented'])

    for r in results:
        fname = r['filename']
        impl = r['our_implementation'] or "-"

        if r['implemented']:
            status = "IMPLEMENTED"
        elif r['compatible']:
            status = "Compatible (not implemented)"
        else:
            status = f"Incompatible: {r['blocking_element_name']}"

        lines.append(f"{fname:<30} {impl:<30} {status:<40}")

    lines.append("-" * 100)
    lines.append(f"Total: {total} | Compatible: {compatible} ({100*compatible/total:.1f}%) | Implemented: {implemented}")

    return "\n".join(lines)


def format_markdown(results: list) -> str:
    """Format results as Markdown table."""
    lines = []

    # Stats
    total = len(results)
    compatible = sum(1 for r in results if r['compatible'])
    implemented = sum(1 for r in results if r['implemented'])

    lines.append("# Falstad Circuit Compatibility Checklist")
    lines.append("")
    lines.append(f"**Total circuits:** {total}")
    lines.append(f"**Compatible:** {compatible} ({100*compatible/total:.1f}%)")
    lines.append(f"**Implemented:** {implemented}")
    lines.append("")

    # Implemented section
    lines.append("## Implemented Circuits")
    lines.append("")
    lines.append("| Falstad File | Our Implementation | Status |")
    lines.append("|--------------|-------------------|--------|")

    for r in results:
        if r['implemented']:
            lines.append(f"| `{r['filename']}` | `{r['our_implementation']}` | IMPLEMENTED |")

    # Compatible but not implemented
    lines.append("")
    lines.append("## Compatible (Not Yet Implemented)")
    lines.append("")
    lines.append("| Falstad File | Status |")
    lines.append("|--------------|--------|")

    for r in results:
        if r['compatible'] and not r['implemented']:
            lines.append(f"| `{r['filename']}` | Compatible |")

    # Incompatible
    lines.append("")
    lines.append("## Incompatible Circuits")
    lines.append("")
    lines.append("| Falstad File | Blocking Element | Element Name |")
    lines.append("|--------------|------------------|--------------|")

    for r in results:
        if not r['compatible']:
            lines.append(f"| `{r['filename']}` | `{r['blocking_element']}` | {r['blocking_element_name']} |")

    return "\n".join(lines)


def format_csv(results: list) -> str:
    """Format results as CSV."""
    lines = []
    lines.append("filename,our_implementation,compatible,implemented,blocking_element,blocking_element_name")

    for r in results:
        lines.append(
            f"{r['filename']},"
            f"{r['our_implementation']},"
            f"{r['compatible']},"
            f"{r['implemented']},"
            f"{r['blocking_element'] or ''},"
            f"{r['blocking_element_name'] or ''}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Falstad circuit compatibility with pyvibrate")
    parser.add_argument("--falstad-dir", type=Path, default=FALSTAD_DIR_DEFAULT,
                        help="Path to Falstad circuits directory")
    parser.add_argument("--output", choices=["table", "markdown", "csv"], default="markdown",
                        help="Output format (default: markdown)")
    parser.add_argument("--save", type=Path, help="Save output to file")

    args = parser.parse_args()

    results = analyze_all_circuits(args.falstad_dir)

    if not results:
        print("No circuits found!", file=sys.stderr)
        sys.exit(1)

    if args.output == "table":
        output = format_table(results)
    elif args.output == "markdown":
        output = format_markdown(results)
    elif args.output == "csv":
        output = format_csv(results)

    if args.save:
        with open(args.save, 'w') as f:
            f.write(output)
        print(f"Output saved to {args.save}")
    else:
        print(output)


if __name__ == "__main__":
    main()
