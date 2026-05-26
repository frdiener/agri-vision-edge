"""
Category helper utilities.

This module intentionally contains no global
dataset/category state anymore.

Dataset semantics should instead originate from:

    DatasetDefinition
"""

from __future__ import annotations


def build_category_map(
    categories,
):
    """
    Build category_id -> category_name map.
    """

    return {

        category["id"]:
            category["name"]

        for category in categories
    }


def build_class_names(
    categories,
):
    """
    Build TFRecord-compatible class name map.
    """

    return {

        category["id"]:
            category["name"].encode("utf-8")

        for category in categories
    }
