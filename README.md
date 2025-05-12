
# Formula Translations Dataset

This dataset contains mathematical formulas with their LaTeX representations, ITL (Intermediate Tree Language) representations, and natural language translations across different domains.

## Dataset Statistics

- Total entries: 217916
- Unique formulas: 217423
- Domains: Aerospace (16822 entries), Automotive/Autonomous Vehicles (16712 entries), Build Pipelines and CI/CD (16743 entries), Financial/Transaction Systems (16769 entries), Home Automation (16753 entries), Industrial Automation/Manufacturing (16787 entries), Medical Devices (16714 entries), Networking/Distributed Systems (16751 entries), Robotics (16801 entries), Security and Authentication (16753 entries), Smart Grid/Energy Management (16719 entries), Version Control and Code Reviews (16823 entries), Web Services/APIs (16769 entries)

## Schema

- **formula_id**: Unique identifier for each formula
- **formula**: The formula text
- **latex**: LaTeX representation of the formula
- **formula_canonical_form**: Canonical form of the formula
- **depth**: Depth of the formula
- **itl_id**: ID of the ITL representation
- **itl_text**: ITL representation text
- **generation_method**: Method used to generate the ITL representation
- **verified**: Whether the ITL representation has been verified
- **is_correct**: Whether the ITL representation is correct
- **translation_id**: ID of the translation
- **domain**: Domain of the natural language translation
- **activity**: Activity associated with the translation
- **translation**: Natural language translation of the formula
- **generation_time**: Time taken to generate the translation

Created on: 2025-05-12
