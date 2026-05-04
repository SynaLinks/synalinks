# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
import copy


def dynamic_enum(schema, prop_to_update, labels, description=None, inline=True):
    """Update a schema with dynamic Enum string.

    Args:
        schema (dict): The schema to update.
        prop_to_update (str): The property to update.
        labels (list): The list of labels (strings).
        description (str, optional): An optional description for the enum.
        inline (bool): If True, write the enum directly into the property
            instead of placing it under `$defs` and pointing to it via a
            `$ref`. Useful for providers whose structured-output backend
            does not resolve `$ref`s strictly (Default to False).

    Returns:
        dict: The updated schema with the enum applied to the specified property.
    """
    schema = copy.deepcopy(schema)
    title = prop_to_update.title().replace("_", "")

    enum_definition = {
        "enum": labels,
        "title": title,
        "type": "string",
    }
    if description:
        enum_definition["description"] = description

    if inline:
        schema.setdefault("properties", {}).update({prop_to_update: enum_definition})
    else:
        if schema.get("$defs"):
            schema = {"$defs": schema.pop("$defs"), **schema}
        else:
            schema = {"$defs": {}, **schema}
        schema["$defs"].update({title: enum_definition})
        schema.setdefault("properties", {}).update(
            {prop_to_update: {"$ref": f"#/$defs/{title}"}}
        )

    return schema


def dynamic_tool_calls(tools, inline=True):
    """
    Generates a dynamic schema for tool calls based on a list of tools.

    This function takes a list of tool objects and constructs a schema that includes
    definitions for each tool's properties, ensuring that each tool call includes a
    "tool_name" field to identify the tool being called.

    Args:
        tools (list): A list of tool objects, each with a name() method and
            an obj_schema() method that returns the schema of the tool.
        inline (bool): If True, embed each per-tool sub-schema directly in
            the `anyOf` branches and skip `$defs`/`$ref` indirection.
            Useful for providers whose structured-output backend does not
            resolve `$ref`s strictly (Default to False).

    Returns:
        (dict): A schema dictionary that defines the structure for tool calls. The schema
            includes definitions for each tool and specifies that tool calls should
            be an array of items, each adhering to one of the tool schemas. The schema
            enforces that the "tool_name" field is required for each tool call.
    """
    tools_schemas_with_tool_names = {}

    for tool in tools:
        tool_name = tool.name
        schema = copy.deepcopy(tool.get_tool_schema())
        if "properties" in schema:
            tool_name_property = {
                "const": tool_name,
                "title": "Tool Name",
                "type": "string",
            }
            new_properties = {"tool_name": tool_name_property, **schema["properties"]}
            schema["properties"] = new_properties

        if "required" in schema:
            required_fields = ["tool_name"]
            required_fields.extend([req for req in schema["required"]])
            schema["required"] = required_fields
        else:
            schema["required"] = ["tool_name"]
        tools_schemas_with_tool_names[schema["title"]] = schema

    tool_names = [tool.name for tool in tools]
    if inline:
        any_of = list(tools_schemas_with_tool_names.values())
    else:
        any_of = [
            {"$ref": "#/$defs/" + schema_key}
            for schema_key in tools_schemas_with_tool_names.keys()
        ]
    tool_calls_schema = {
        "additionalProperties": False,
        "properties": {
            "tool_calls": {
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "enum": tool_names,
                            "type": "string",
                            "title": "Tool Name",
                        }
                    },
                    "required": ["tool_name"],
                    "anyOf": any_of,
                },
                "title": "Tool Calls",
                "type": "array",
            }
        },
        "required": ["tool_calls"],
        "title": "ToolCalls",
        "type": "object",
    }
    if not inline:
        tool_calls_schema = {
            "$defs": tools_schemas_with_tool_names,
            **tool_calls_schema,
        }

    return tool_calls_schema


def dynamic_tool_choice(tools, inline=True):
    tools_schemas_with_tool_names = {}

    for tool in tools:
        tool_name = tool.name
        schema = copy.deepcopy(tool.get_tool_schema())
        if "properties" in schema:
            tool_name_property = {
                "const": tool_name,
                "title": "Tool Name",
                "type": "string",
            }
            new_properties = {"tool_name": tool_name_property, **schema["properties"]}
            schema["properties"] = new_properties

        if "required" in schema:
            required_fields = ["tool_name"]
            required_fields.extend([req for req in schema["required"]])
            schema["required"] = required_fields
        else:
            schema["required"] = ["tool_name"]
        tools_schemas_with_tool_names[schema["title"]] = schema

    tool_names = [tool.name for tool in tools]
    if inline:
        any_of = list(tools_schemas_with_tool_names.values())
    else:
        any_of = [
            {"$ref": "#/$defs/" + schema_key}
            for schema_key in tools_schemas_with_tool_names.keys()
        ]
    tool_choice_schema = {
        "additionalProperties": False,
        "properties": {
            "tool_choice": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "enum": tool_names,
                        "type": "string",
                        "title": "Tool Name",
                    }
                },
                "required": ["tool_name"],
                "anyOf": any_of,
                "title": "Tool Choice",
            }
        },
        "required": ["tool_choice"],
        "title": "ToolChoice",
        "type": "object",
    }
    if not inline:
        tool_choice_schema = {
            "$defs": tools_schemas_with_tool_names,
            **tool_choice_schema,
        }

    return tool_choice_schema
