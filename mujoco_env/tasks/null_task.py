from gymnasium.core import ActType

from ..common.defs.types import InfoDict
from .task import Task


class NullTask(Task):
    """A task with no goals or objectives. A NullTask is never complete and always rewards 0"""

    def reset(self, *args, **kwargs) -> None:
        pass

    def begin_frame(self, action: ActType) -> None:
        pass

    def end_frame(self, action: ActType) -> None:
        pass

    def score(self) -> float:
        return 0.0

    def is_done(self) -> bool:
        return False

    def get_info(self) -> InfoDict:
        return {}

    def update_render(self, viewer) -> None:
        pass
