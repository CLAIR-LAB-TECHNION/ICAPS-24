from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import count


@dataclass(frozen=True)
class Spec:
    id: int = field(default_factory=count().__next__, init=False)

    def __index__(self):
        return self.id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict={}):
        cpy = self.copy()
        memodict[id(self)] = cpy
        return cpy

    def copy(self):
        return replace(self)