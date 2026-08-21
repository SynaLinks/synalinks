<!-- colab-badge:start -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SynaLinks/synalinks/blob/main/notebooks/guides/graphrag.ipynb)
<!-- colab-badge:end -->

::: guides.32_graphrag

## Source

````python
--8<-- "guides/32_graphrag.py:source"
````

## Run log

This guide calls `synalinks.enable_logging(log_level="info")`, so a full run
traces every module call: the graph load, community building, both retrievers,
and the map-reduce generators. The log below is the unedited output of running
the guide above with local models.

??? example "Full run log: `guides/32_graphrag.log`"

    ```text
    --8<-- "guides/32_graphrag.log"
    ```
