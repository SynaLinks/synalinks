from synalinks.src.backend.config import api_key
from synalinks.src.backend.config import backend
from synalinks.src.backend.config import epsilon
from synalinks.src.backend.config import floatx

if backend() == "pydantic":
    import pydantic

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common import name_scope
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum
from synalinks.src.backend.common.dynamic_json_schema_utils import dynamic_enum_array
from synalinks.src.backend.common.json_schema_utils import concatenate_schema
from synalinks.src.backend.common.json_schema_utils import contains_schema
from synalinks.src.backend.common.json_schema_utils import factorize_schema
from synalinks.src.backend.common.json_schema_utils import in_mask_schema
from synalinks.src.backend.common.json_schema_utils import is_schema_equal
from synalinks.src.backend.common.json_schema_utils import out_mask_schema
from synalinks.src.backend.common.json_schema_utils import prefix_schema
from synalinks.src.backend.common.json_schema_utils import standardize_schema
from synalinks.src.backend.common.json_schema_utils import suffix_schema
from synalinks.src.backend.common.json_utils import concatenate_json
from synalinks.src.backend.common.json_utils import factorize_json
from synalinks.src.backend.common.json_utils import in_mask_json
from synalinks.src.backend.common.json_utils import out_mask_json
from synalinks.src.backend.common.json_utils import prefix_json
from synalinks.src.backend.common.json_utils import suffix_json
from synalinks.src.backend.common.stateless_scope import StatelessScope
from synalinks.src.backend.common.stateless_scope import get_stateless_scope
from synalinks.src.backend.common.stateless_scope import in_stateless_scope
from synalinks.src.backend.common.symbolic_data_model import any_symbolic_data_models
from synalinks.src.backend.common.symbolic_data_model import is_symbolic_data_model
from synalinks.src.backend.common.symbolic_scope import SymbolicScope

if backend() == "pydantic":
    from pydantic import Field

    from synalinks.src.backend.pydantic.base import ChatMessage
    from synalinks.src.backend.pydantic.base import ChatMessages
    from synalinks.src.backend.pydantic.base import ChatRole
    from synalinks.src.backend.pydantic.base import EmbeddedEntity
    from synalinks.src.backend.pydantic.base import Embedding
    from synalinks.src.backend.pydantic.base import Embeddings
    from synalinks.src.backend.pydantic.base import Entities
    from synalinks.src.backend.pydantic.base import Entity
    from synalinks.src.backend.pydantic.base import GenericInputs
    from synalinks.src.backend.pydantic.base import GenericIO
    from synalinks.src.backend.pydantic.base import GenericOutputs
    from synalinks.src.backend.pydantic.base import GenericResult
    from synalinks.src.backend.pydantic.base import Instructions
    from synalinks.src.backend.pydantic.base import KnowledgeGraph
    from synalinks.src.backend.pydantic.base import Prediction
    from synalinks.src.backend.pydantic.base import Relation
    from synalinks.src.backend.pydantic.base import Relations
    from synalinks.src.backend.pydantic.base import SimilaritySearch
    from synalinks.src.backend.pydantic.base import TripletSearch
    from synalinks.src.backend.pydantic.base import is_chat_message
    from synalinks.src.backend.pydantic.base import is_chat_messages
    from synalinks.src.backend.pydantic.base import is_embedded_entity
    from synalinks.src.backend.pydantic.base import is_embedding
    from synalinks.src.backend.pydantic.base import is_embeddings
    from synalinks.src.backend.pydantic.base import is_entities
    from synalinks.src.backend.pydantic.base import is_entity
    from synalinks.src.backend.pydantic.base import is_instructions
    from synalinks.src.backend.pydantic.base import is_knowledge_graph
    from synalinks.src.backend.pydantic.base import is_prediction
    from synalinks.src.backend.pydantic.base import is_relation
    from synalinks.src.backend.pydantic.base import is_relations
    from synalinks.src.backend.pydantic.base import is_similarity_search
    from synalinks.src.backend.pydantic.base import is_triplet_search
    from synalinks.src.backend.pydantic.core import IS_THREAD_SAFE
    from synalinks.src.backend.pydantic.core import DataModel as BackendDataModel
    from synalinks.src.backend.pydantic.core import any_data_model
    from synalinks.src.backend.pydantic.core import any_meta_class
    from synalinks.src.backend.pydantic.core import is_data_model
    from synalinks.src.backend.pydantic.core import is_meta_class
    from synalinks.src.backend.pydantic.module import PydanticModule
else:
    raise ValueError(f"Unable to import backend : {backend()}")

from synalinks.src.backend.common.json_data_model import JsonDataModel
from synalinks.src.backend.common.json_data_model import is_json_data_model
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.backend.common.variables import Variable


@synalinks_export(["synalinks.DataModel", "synalinks.backend.DataModel"])
class DataModel(BackendDataModel):  # noqa: F811
    pass


backend_name_scope = name_scope  # noqa: F405


@synalinks_export("synalinks.name_scope")
class name_scope(backend_name_scope):
    pass


@synalinks_export("synalinks.ops.convert_to_json_data_model")
def convert_to_json_data_model(x):
    return JsonDataModel(schema=x.get_schema(), json=x.get_json()) if x else x


@synalinks_export("synalinks.ops.convert_to_symbolic_data_model")
def convert_to_symbolic_data_model(x):
    return SymbolicDataModel(schema=x.get_schema()) if x else x


async def compute_output_spec(fn, *args, **kwargs):
    """Computes the output specification of a function.

    This function wraps the given function call in a stateless and symbolic scope
    to compute the output specification.

    Args:
        fn (callable): The function to compute the output specification for.
        *args (positional arguments): The positional arguments to pass to the function.
        **kwargs (keyword arguments): The keyword arguments to pass to the function.

    Returns:
        (SymbolicDataModel): The output specification of the function.
    """
    with StatelessScope(), SymbolicScope():
        output_spec = await fn(*args, **kwargs)
    return output_spec
