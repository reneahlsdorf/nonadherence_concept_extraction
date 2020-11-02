"""This module provides some helpers for registering (pipeline stage) modules which can be controlled via the CLI."""

"""
Medical lexicon NLP extraction pipeline

File contains: Helper classes for the overall framework


-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

from abc import ABCMeta, abstractmethod, abstractclassmethod
import re

class ModuleMetaclass(ABCMeta):
    """MetaClass which remembers all (pipeline stage) modules
    """

    __all_modules__ = dict()

    def __new__(cls, name, parents, dct):
        inst = super(ModuleMetaclass, cls).__new__(cls, name, parents, dct)
        if name != 'Module':
            cls.__all_modules__[inst.module_name()] = inst
        return inst

    @classmethod
    def get_modules(cls):
        """Returns a list of all public modules.

        Returns:
            dict: Dict of all public modules.
        """
        return cls.__all_modules__


class Module(metaclass=ModuleMetaclass):
    """The base class for a (pipeline stage) module.
    """

    @classmethod
    def module_name(cls):
        """Returns the name of this module in snake case format.

        Returns:
            str: Module Name
        """
        return cls.camelcase_to_snakecase(cls.__name__).replace('_module', '').replace('module', '')

    @abstractclassmethod
    def register_argparser_object(cls, subparser_instance):
        """Register this module to the pipeline tool CLI

        Args:
            subparser (argparse.ArgumentParser): The argparse subparser to add this module to.
        """
        raise NotImplementedError()

    @staticmethod
    def camelcase_to_snakecase(name):
        """Converts string from camel to snake case

        Args:
            name (str): Camel-casted string

        Returns:
            str: Snake-cased string
        """
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


            
