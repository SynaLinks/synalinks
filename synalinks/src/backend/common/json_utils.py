# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import collections
import copy
import re

from synalinks.src.utils.nlp_utils import add_suffix
from synalinks.src.utils.nlp_utils import is_plural
from synalinks.src.utils.nlp_utils import to_plural_without_numerical_suffix
from synalinks.src.utils.nlp_utils import to_singular_without_numerical_suffix


def prefix_json(json, prefix):
    """Add a prefix to the json object keys"""
    json = copy.deepcopy(json)
    prefixed_json = {}
    for prop_key, prop_value in json.items():
        prefixed_json[f"{prefix}_{prop_key}"] = prop_value
    return prefixed_json


def suffix_json(json, suffix):
    """Add a suffix to the json object keys"""
    json = copy.deepcopy(json)
    suffixed_json = {}
    for prop_key, prop_value in json.items():
        suffixed_json[f"{prop_key}_{suffix}"] = prop_value
    return suffixed_json


def concatenate_json(json1, json2):
    """Concatenate two Json objects into a single schema.

    This function merges the properties of two Json object into a single object.
    If there are conflicting property names, it appends a suffix to make them unique.

    Args:
        json1 (dict): The first Json object to be concatenated.
        json2 (dict): The second Json object to be concatenated.

    Returns:
        (dict): A new Json object that combines the properties of the input objects.
    """
    json1 = copy.deepcopy(json1)
    json2 = copy.deepcopy(json2)

    result_json = {}

    def add_property(prop_key, prop_value, suffix=0):
        new_prop_key = prop_key
        original_value = prop_value
        while new_prop_key in result_json:
            suffix += 1
            new_prop_key = add_suffix(prop_key, suffix)
        result_json[new_prop_key] = original_value

    for prop_key, prop_value in json1.items():
        add_property(prop_key, prop_value)

    for prop_key, prop_value in json2.items():
        add_property(prop_key, prop_value)

    return result_json


def factorize_json(json):
    """Factorize a Json object by grouping similar properties into lists.

    This function groups similar properties in a Json object into list properties.
    It identifies similar properties based on their base names
    and creates array for them.

    Args:
        json (dict): The input Json object to factorize.

    Returns:
        (dict): A factorized Json object with grouped properties.
    """
    json = copy.deepcopy(json)
    # Initialize the resulting Json object
    result_json = {}

    for prop_key, prop_value in json.items():
        # Get the base name
        base_key = to_singular_without_numerical_suffix(prop_key)
        plural_key = to_plural_without_numerical_suffix(base_key)

        # Find all similar properties
        similar_props = [
            p
            for p in json.keys()
            if to_singular_without_numerical_suffix(p) == base_key and p != prop_key
        ]
        if similar_props and not is_plural(prop_key):
            if plural_key not in result_json:
                # Create an array property
                result_json[plural_key] = []

            # Add the values to the list
            if isinstance(prop_value, list):
                result_json[plural_key].extend(prop_value)
            else:
                result_json[plural_key].append(prop_value)
        else:
            if not is_plural(prop_key):
                result_json[base_key] = prop_value
            else:
                # If the property is a plural (a list) ensure it is added to the result
                result_json[prop_key] = prop_value

    return result_json


def decompose_json(json):
    """Decompose a Json object by expanding list properties into individual ones.

    This is the inverse of factorize_json. It takes list properties and
    expands them into individual properties with numerical suffixes.
    For example `foos: ["a", "b"]` becomes `foo: "a", foo_1: "b"`.

    Args:
        json (dict): The input Json object to decompose.

    Returns:
        (dict): A decomposed Json object with expanded properties.
    """
    json = copy.deepcopy(json)
    result_json = {}

    for prop_key, prop_value in json.items():
        if is_plural(prop_key) and isinstance(prop_value, list):
            singular_key = to_singular_without_numerical_suffix(prop_key)
            for i, item in enumerate(prop_value):
                if i == 0:
                    result_json[singular_key] = item
                else:
                    result_json[add_suffix(singular_key, i)] = item
        else:
            result_json[prop_key] = prop_value

    return result_json


def out_mask_json(json, mask=None, pattern=None, recursive=True):
    """Mask specific fields of a Json object.

    This function look for properties to mask and remove them.
    It ignores the suffixes that other operations could add.

    Args:
        json (dict): The input Json object to mask.
        mask (list): The base key list to remove.
        pattern (str): Optional. A regex pattern to match property keys
            to remove. If provided, properties whose base key matches
            the pattern will be removed.
        recursive (bool): Whether or not to remove
            recursively for nested objects (default True).

    Returns:
        (dict): A masked Json object with removed properties.
    """
    json = copy.deepcopy(json)

    if not mask and not pattern:
        return json

    if mask:
        # Ensure that the mask keys are in singular form
        mask = [to_singular_without_numerical_suffix(k) for k in mask]

    compiled_pattern = re.compile(pattern) if pattern else None

    stack = collections.deque([json])

    while stack:
        current = stack.pop()
        keys_to_delete = []

        for prop_key, prop_value in current.items():
            base_key = to_singular_without_numerical_suffix(prop_key)

            if mask and base_key in mask:
                keys_to_delete.append(prop_key)
            elif compiled_pattern and compiled_pattern.search(base_key):
                keys_to_delete.append(prop_key)

            if recursive:
                if isinstance(prop_value, dict):
                    stack.append(prop_value)
                elif isinstance(prop_value, list):
                    for item in prop_value:
                        if isinstance(item, dict):
                            stack.append(item)

        for key in keys_to_delete:
            del current[key]

    return json


def in_mask_json(json, mask=None, pattern=None, recursive=True):
    """Keep specific fields of a Json object.

    This function looks for properties to keep and removes all others.
    It ignores the suffixes that other operations could add.

    Args:
        json (dict): The input Json object to mask.
        mask (list): The base key list to keep.
        pattern (str): Optional. A regex pattern to match property keys
            to keep. If provided, properties whose base key matches
            the pattern will be kept.
        recursive (bool): Whether or not to keep
            recursively for nested objects (default True).

    Returns:
        (dict): A masked Json object with only the specified properties.
    """
    json = copy.deepcopy(json)

    if not mask and not pattern:
        return {}

    if mask:
        # Ensure that the mask keys are in singular form
        mask = [to_singular_without_numerical_suffix(k) for k in mask]

    compiled_pattern = re.compile(pattern) if pattern else None

    stack = collections.deque([json])

    while stack:
        current = stack.pop()
        keys_to_keep = []

        for prop_key, prop_value in current.items():
            base_key = to_singular_without_numerical_suffix(prop_key)

            if mask and base_key in mask:
                keys_to_keep.append(prop_key)
            elif compiled_pattern and compiled_pattern.search(base_key):
                keys_to_keep.append(prop_key)

            if recursive:
                if isinstance(prop_value, dict):
                    stack.append(prop_value)
                elif isinstance(prop_value, list):
                    keys_to_keep.append(prop_key)
                    for item in prop_value:
                        if isinstance(item, dict):
                            stack.append(item)

        keys_to_delete = set(current.keys()) - set(keys_to_keep)
        for key in keys_to_delete:
            del current[key]

    return json
