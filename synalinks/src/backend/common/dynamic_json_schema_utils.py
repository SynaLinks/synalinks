# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)
import copy


def dynamic_enum(schema, prop_to_update, labels, description=None):
    """Update a schema with dynamic Enum string.

    Args:
        schema (dict): The schema to update.
        prop_to_update (str): The property to update.
        labels (list): The list of labels (strings).
        description (str, optional): An optional description for the enum.

    Returns:
        dict: The updated schema with the enum applied to the specified property.
    """
    schema = copy.deepcopy(schema)
    if schema.get("$defs"):
        schema = {"$defs": schema.pop("$defs"), **schema}
    else:
        schema = {"$defs": {}, **schema}
    title = prop_to_update.title().replace("_", "")

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

    schema["$defs"].update({title: enum_definition})

    schema.setdefault("properties", {}).update(
        {prop_to_update: {"$ref": f"#/$defs/{title}"}}
    )

    return schema


def dynamic_tool_calls(tools):
    """
    Generates a dynamic schema for tool calls based on a list of tools.

    This function takes a list of tool objects and constructs a schema that includes
    definitions for each tool's properties, ensuring that each tool call includes a
    "tool_name" field to identify the tool being called.

    Args:
        tools (list): A list of tool objects, each with a name() method and
            an obj_schema() method that returns the schema of the tool.

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

    tool_calls_schema = {
        "$defs": tools_schemas_with_tool_names,
        "additionalProperties": False,
        "properties": {
            "tool_calls": {
                "items": {
                    "anyOf": [
                        {"$ref": "#/$defs/" + schema_key}
                        for schema_key in tools_schemas_with_tool_names.keys()
                    ]
                },
                "title": "Tool Calls",
                "type": "array",
            }
        },
        "required": ["tool_calls"],
        "title": "ToolCalls",
        "type": "object",
    }

    return tool_calls_schema


def dynamic_tool_choice(tools):
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

    tool_choice_schema = {
        "$defs": tools_schemas_with_tool_names,
        "additionalProperties": False,
        "properties": {
            "tool_choice": {
                "anyOf": [
                    {"$ref": "#/$defs/" + schema_key}
                    for schema_key in tools_schemas_with_tool_names.keys()
                ],
                "title": "Tool Choice",
            }
        },
        "required": ["tool_choice"],
        "title": "ToolChoice",
        "type": "object",
    }

    return tool_choice_schema
