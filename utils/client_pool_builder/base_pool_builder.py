from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseClientPoolBuilder(ABC):
    @abstractmethod
    def build_pool(self, 
                   n_client: int,
                   client_config: Dict[str, Any],
                   **kwargs) -> List[Dict[str, Any]]:
        pass
