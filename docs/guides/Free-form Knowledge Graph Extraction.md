<!-- colab-badge:start -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SynaLinks/synalinks/blob/main/notebooks/guides/free_form_knowledge_graph_extraction.ipynb)
<!-- colab-badge:end -->

::: guides.28_free_form_knowledge_graph_extraction

## Source

````python
--8<-- "guides/28_free_form_knowledge_graph_extraction.py:source"
````

## Run log

This guide calls `synalinks.enable_logging(log_level="info")`, so a full run
traces every module call — entity/relation extraction, the embedding step, and
the graph update. The log below is the output of running the guide above with
local models; the per-entity **embedding vectors (1024 floats each) are
collapsed** to a placeholder to keep it readable — everything else is unedited.

??? example "Full run log — `guides/28_free_form_knowledge_graph_extraction.log`"

    ```text
    --8<-- "guides/28_free_form_knowledge_graph_extraction.log"
    ```
