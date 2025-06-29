"""
Monkey patch for ipapy to fix collections import issue in Python 3.10+.

This patch must be imported before any ipapy imports to work correctly.
"""

import sys
import collections.abc

# Monkey patch collections to include MutableSequence from collections.abc
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence

# Also patch other potentially missing ABCs that might be used
# if not hasattr(collections, 'Sequence'):
#     collections.Sequence = collections.abc.Sequence

# if not hasattr(collections, 'Iterable'):
#     collections.Iterable = collections.abc.Iterable

# if not hasattr(collections, 'Container'):
#     collections.Container = collections.abc.Container

# if not hasattr(collections, 'Sized'):
#     collections.Sized = collections.abc.Sized

# if not hasattr(collections, 'Callable'):
#     collections.Callable = collections.abc.Callable

# if not hasattr(collections, 'Set'):
#     collections.Set = collections.abc.Set

# if not hasattr(collections, 'MutableSet'):
#     collections.MutableSet = collections.abc.MutableSet

# if not hasattr(collections, 'Mapping'):
#     collections.Mapping = collections.abc.Mapping

# if not hasattr(collections, 'MutableMapping'):
#     collections.MutableMapping = collections.abc.MutableMapping
