from typing import Dict, Type, Optional, List
from pipelines.base_pipeline import BaseGenrePipeline

class PipelineRegistry:
    """
    Central registry to enable extensible plugin-based pipelines.
    """
    _registry: Dict[str, Type[BaseGenrePipeline]] = {}

    @classmethod
    def register(cls, name: str, pipeline_class: Type[BaseGenrePipeline]):
        cls._registry[name] = pipeline_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseGenrePipeline]]:
        return cls._registry.get(name)

    @classmethod
    def list_pipelines(cls) -> List[str]:
        return list(cls._registry.keys())
