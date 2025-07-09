"""
Monkey patch for ipapy to fix collections import issue in Python 3.10+.

This patch must be imported before any ipapy imports to work correctly.
"""

import sys
import collections.abc

# Monkey patch collections to include MutableSequence from collections.abc
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence
