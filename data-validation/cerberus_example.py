#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ─────────────────────────────────────────────────────────────────────────────
# DEVELOPER NOTES
# ─────────────────────────────────────────────────────────────────────────────

"""
Module Overview
───────────────
Reusable validation schemas using Cerberus for common scalar, string,
datetime, and nested data structures. Intended as copy-paste templates
for consistent variable integrity checks.

Dependencies
────────────
- cerberus >= 1.3

Limitations
───────────
- Tested with Python 3.11.
- Assumes UTF-8 encoding.
- Float regex precision is not natively supported by Cerberus and would
  require a custom validator.
"""

__author__  = "Aidan Calderhead"
__created__ = "2025-10-03"
__license__ = "MIT"

from cerberus import Validator  # type: ignore
from pprint   import pprint

# ─────────────────────────────────────────────────────────────────────────────
# COMMON SCALAR CHECKS
# ─────────────────────────────────────────────────────────────────────────────
schema_scalars = {
    "positive_integer": {
        "type": "integer",
        "min": 1
    },
    "bounded_integer": {
        "type": "integer",
        "min": 0,
        "max": 100
    },
    # NOTE: Cerberus does not support regex on floats; either store as string
    # or use a custom validator.
    "floating_precision": {
        "type": "string",
        "regex": r"^\d+(\.\d{1,2})?$"
    },
    "boolean_required_true": {
        "type": "boolean",
        "allowed": [True]
    },
    "boolean_required_false": {
        "type": "boolean",
        "allowed": [False]
    },
    "nullable_field": {
        "type": "string",
        "nullable": True
    },
    "non_nullable_field": {
        "type": "string",
        "nullable": False
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# STRING / SEQUENCE CHECKS
# ─────────────────────────────────────────────────────────────────────────────
schema_strings = {
    "short_code": {
        "type": "string",
        "minlength": 2,
        "maxlength": 5,
        "regex": "^[A-Z]+$"
    },
    "email": {
        "type": "string",
        "regex": r"^[\w\.-]+@[\w\.-]+\.\w+$"
    },
    "enum_choice": {
        "type": "string",
        "allowed": ["low", "medium", "high"]
    },
    "list_of_ints": {
        "type": "list",
        "schema": {"type": "integer"}
    },
    "fixed_length_list": {
        "type": "list",
        "minlength": 3,
        "maxlength": 3,
        "schema": {"type": "string"}
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# DATETIME CHECKS
# ─────────────────────────────────────────────────────────────────────────────
schema_datetime = {
    "past_date": {
        "type": "datetime",
        "max": "now"
    },
    "future_date": {
        "type": "datetime",
        "min": "now"
    },
    "bounded_datetime": {
        "type": "datetime",
        "min": "2020-01-01T00:00:00",
        "max": "2030-12-31T23:59:59"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# NESTED DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────
schema_nested = {
    "user": {
        "type": "dict",
        "schema": {
            "id": {"type": "integer", "min": 1},
            "name": {"type": "string", "minlength": 1},
            "email": {"type": "string", "regex": r".+@.+"},
        }
    },
    "orders": {
        "type": "list",
        "schema": {
            "type": "dict",
            "schema": {
                "order_id": {"type": "integer"},
                "amount": {"type": "float", "min": 0},
            }
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USE
# ─────────────────────────────────────────────────────────────────────────────
data = {
    "positive_integer":      5,
    "bounded_integer":       42,
    "floating_precision":    "3.14",
    "boolean_required_true": True,
    "email":     "test@example.com",
    "past_date": "2022-01-01T00:00:00",
    "user":      {"id": 1, "name": "Alice", "email": "alice@example.com"},
    "orders":    [{"order_id": 1, "amount": 99.99}]
}

schema = {**schema_scalars, **schema_strings, **schema_datetime, **schema_nested}
v = Validator(schema, require_all=True)
pprint(v.validate(data))      # True or False
pprint(v.errors)              # Show detailed failures
