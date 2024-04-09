from __future__ import annotations

from importlib import import_module
from typing import Iterable, Type, TypeVar, Optional, Any, Callable

import numpy as np

from .defs.types import Entrypoint

ENTRYPOINT_MODULE_FUNC_SEP = ':'

T = TypeVar('T')


def arraylike_func(func: Callable[..., T]) -> Callable[..., T]:
    """
    A decorator for functions that accept array-like objects as arguments. The decorator converts the arguments to numpy
    arrays before passing them to the function.
    :param func: the function to be decorated.
    :return: the decorated function. identical to `func` but converts the positional arguments to numpy arrays.
    """
    def with_arraylike(*args, **kwargs):
        args = [np.asarray(arg) for arg in args]
        return func(*args, **kwargs)

    return with_arraylike


def nested_dict_update(src, dest, inplace=False):
    """
    Recursively updates a dictionary with another dictionary. If a key in `src` exists in `dest` and the value of that
    key is a dictionary, the function is called recursively on the two dictionaries. Otherwise, the value of the key in
    `dest` is overwritten by the value of the key in `src`. Missing keys in `dest` are added.
    :param src: The dictionary with update values
    :param dest: The dictionary to update
    :return:
    """
    if not inplace:
        dest = dest.copy()

    for k, v in src.items():
        dest_v = dest.get(k, {})
        if isinstance(v, dict) and isinstance(dest_v, dict):
            dest[k] = nested_dict_update(v, dest_v)
        else:
            dest[k] = v

    return dest


def load_from_entrypoint(entrypoint: Entrypoint):
    """
    Loads an exported value from a given module described as an entrypoint string. The entrypoint string is of the form
    `module_name:function_name`. For example, we can load function `foo` from module `my_module.py` in package
    `my_package` with entrypoint "my_package.my_module:foo".

    >>> load_from_entrypoint("math:pi")
    3.141592653589793
    >>> load_from_entrypoint("math:ceil")(2.3)
    3

    :param entrypoint: A string of the form `path.to.module:class_or_function`.
    :return: The function specified by the entrypoint string.
    """
    # split the module path from the name of the exported value
    module_path, exported_name = entrypoint.split(ENTRYPOINT_MODULE_FUNC_SEP)

    # dynamically import the module
    module = import_module(module_path)

    # get the desired exported value
    exported = getattr(module, exported_name)

    return exported


def set_iterable_arg(cls: Type[T], arg: Optional[Iterable[T | Any] | T | Any]) -> tuple[T, ...]:
    """
    Converts an argument to a tuple of a given type. Enables using alternative initialization arguments that must
    be iterable, immutable, and may contain a mix of values.
    :param cls: the type of values in the output tuple.
    :param arg: a single value or iterable of values to be converted to a tuple of type `cls`.
    :return: a tuple of the same type as `cls`.
    """
    # None input gives an empty immutable tuple
    if arg is None:
        return tuple()

    # Force arg to be a list
    if not isinstance(arg, Iterable) or isinstance(arg, str):
        arg = [arg]
    else:
        arg = list(arg)

    # update list items as `cls` types if needed
    for i, item in enumerate(arg):
        if not isinstance(item, cls):
            arg[i] = cls(item)

    # set arg to immutable tuple type
    return tuple(arg)
