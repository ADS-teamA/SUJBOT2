#!/usr/bin/env python3
"""Extract Input schema classes from tier2_advanced.py"""

import subprocess
import re

# Get tier2_advanced.py from main branch
result = subprocess.run(
    ["git", "show", "main:src/agent/tools/tier2_advanced.py"],
    capture_output=True,
    text=True,
    cwd="/home/prusemic/SUJBOT2"
)

content = result.stdout
lines = content.split('\n')

# Extract each Input class
input_classes = {}
current_class = None
class_lines = []
indent_level = 0

for i, line in enumerate(lines):
    # Detect class definition
    if re.match(r'^class \w+Input\(ToolInput\):', line):
        # Save previous class
        if current_class:
            input_classes[current_class] = '\n'.join(class_lines)

        # Start new class
        current_class = line.split('(')[0].replace('class ', '').strip()
        class_lines = [line]
        indent_level = len(line) - len(line.lstrip())

    # Collect class body (indented lines after class definition)
    elif current_class and line and not line[0].isspace():
        # End of class (next top-level definition)
        if line.startswith('class ') or line.startswith('@') or line.startswith('def '):
            input_classes[current_class] = '\n'.join(class_lines)
            current_class = None
            class_lines = []

    # Add line to current class
    elif current_class:
        class_lines.append(line)

# Save last class
if current_class:
    input_classes[current_class] = '\n'.join(class_lines)

# Print mapping
print("# Input Classes Extracted:")
print(f"# Total: {len(input_classes)}\n")

for name, code in input_classes.items():
    print(f"\n{'='*80}")
    print(f"# {name}")
    print('='*80)
    print(code)
