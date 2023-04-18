"""Abstract Wrapper class for the some molecule defined in the specific definitions in terms of
the """
import abc
from abc import ABC

from typing import Tuple
from nptyping import NDArray


class AbstractMolecule(ABC):
    name: str = ""
    num_orbitals: int = 0
    num_electrons: int = 0

    @abc.abstractmethod
    def get_integral_matrices(self) -> Tuple[NDArray, NDArray]:
        pass
