import random
from typing import Callable, Iterable, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class LazyDataIterable(Iterable[T]):
    def __init__(self, base_iterable_func: Callable[[], Iterator[T]]):
        self.__base_iterable_func = base_iterable_func

    def __iter__(self) -> Iterator[T]:
        return self.__base_iterable_func()


class MemorizedDataIterable(Iterable[T]):
    def __init__(self, base_iterable_func: Callable[[], Iterator[T]], shuffle: bool = False):
        self.__base_iterable_func = base_iterable_func
        self.__elements: List[T] = []
        self.__use_cache: bool = False
        self.__shuffle = shuffle

    def __yield_and_store(self, base: Iterator[T]):
        for element in base:
            self.__elements.append(element)
            yield element
        self.__use_cache = True
        del self.__base_iterable_func

    def __iter__(self) -> Iterator[T]:
        if self.__use_cache:
            if self.__shuffle:
                random.shuffle(self.__elements)
            return iter(self.__elements)
        else:
            return self.__yield_and_store(self.__base_iterable_func())

    def __call__(self) -> Iterator[T]:
        return iter(self)


def enforce_not_None(e: Optional[T]) -> T:
    """Enforce non-nullness of input. Used for typechecking and runtime safety."""
    if e is None:
        raise Exception("Input is None.")
    return e
