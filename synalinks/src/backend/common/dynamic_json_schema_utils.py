# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)
import copy

from synalinks.src.utils.naming import to_snake_case
from synalinks.src.utils.nlp_utils import to_plural_property


def dynamic_enum(schema, prop_to_update, labels, parent_schema=None, description=None):
    """Update a schema with dynamic Enum string.

    Args:
        schema (dict): The schema to update.
        prop_to_update (str): The property to update.
        labels (list): The list of labels (strings).
        parent_schema (dict, optional): An optional parent schema to use as the base.
        description (str, optional): An optional description for the enum.

    Returns:
        dict: The updated schema with the enum applied to the specified property.
    """
    schema = copy.deepcopy(schema)
    if schema.get("$defs"):
        schema = {"$defs": schema.pop("$defs"), **schema}
    else:
        schema = {"$defs": {}, **schema}
    if parent_schema:
        parent_schema = copy.deepcopy(parent_schema)
    title = prop_to_update.title().replace("_", " ")

    if description:
        enum_definition = {
            "enum": labels,
            "description": description,
            "title": title,
            "type": "string",
        }
    else:
        enum_definition = {
            "enum": labels,
            "title": title,
            "type": "string",
        }

    if parent_schema:
        parent_schema["$defs"].update({title: enum_definition})
    else:
        schema["$defs"].update({title: enum_definition})

    schema.setdefault("properties", {}).update(
        {prop_to_update: {"$ref": f"#/$defs/{title}"}}
    )

    return parent_schema if parent_schema else schema


def dynamic_enum_array(schema, prop_to_update, labels, parent_schema=None, description=None):
    """Update a schema with dynamic Enum list for array properties.

    This function takes a schema with an array property and constrains the items
    in that array to be from a specific enum of labels.

    Args:
        schema (dict): The schema to update (should contain an array property).
        prop_to_update (str): The array property to update with enum constraints.
        labels (list): The list of labels (strings) for the enum.
        parent_schema (dict, optional): An optional parent schema to use as the base.
        description (str, optional): An optional description for the enum.

    Returns:
        dict: The updated schema with the enum applied to the array items.
    """
    schema = copy.deepcopy(schema)
    
    # Ensure $defs is at the beginning of the schema
    if schema.get("$defs"):
        schema = {"$defs": schema.pop("$defs"), **schema}
    else:
        schema = {"$defs": {}, **schema}
    
    if parent_schema:
        parent_schema = copy.deepcopy(parent_schema)
    
    enum_title = to_singular_property(prop_to_update.title()).replace("_", "")

    # Create the enum definition
    if description:
        enum_definition = {
            "enum": labels,
            "description": description,
            "title": enum_title,
            "type": "string",
        }
    else:
        enum_definition = {
            "enum": labels,
            "title": enum_title,
            "type": "string",
        }

    if parent_schema:
        parent_schema["$defs"].update({title: enum_definition})
    else:
        schema["$defs"].update({title: enum_definition})

    schema.setdefault("properties", {}).update({
        prop_to_update: {
            "items": {
                "$ref": f"#/$defs/{enum_title}"
            },
            "type": "array",
            "minItems": 1,
            "uniqueItems": True  # TODO: may unlock multiple function calls of the same type?
        }
    })

    return parent_schema if parent_schema else schema


def dynamic_list(schema):
    """Update a schema to convert it to a nested list"""
    schema = copy.deepcopy(schema)
    if schema.get("$defs"):
        defs = schema.pop("$defs")
    else:
        defs = {}
    title = schema.get("title")
    property_list = to_plural_property(to_snake_case(title))
    title_list = "".join([word.capitalize() for word in property_list.split("_")])
    new_schema = {
        "$defs": {title: schema, **defs},
        "additionalProperties": False,
        "properties": {
            property_list: {
                "items": {
                    "$ref": f"#/$defs/{title}",
                },
                "title": title_list,
                "type": "array",
            }
        },
        "required": [property_list],
        "title": title_list,
        "type": "object",
    }
    return new_schema
