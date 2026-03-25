"""
core/constants.py

System-wide structural constants.
These are fixed architecture rules, not runtime config values.
"""

# Portfolio structure
MAX_HOLDINGS = 3

# Daily entry limits
MAX_STRONG_ENTRIES_PER_DAY = 1
MAX_GENERAL_ENTRIES_PER_DAY = 1
MAX_REPLACEMENTS_PER_DAY = 1

# Replacement behavior
REPLACEMENT_REDUCTION_RATIO = 0.5