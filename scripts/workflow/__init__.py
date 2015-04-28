"""Workflow package."""

import os
import inspect
import json
from collections import OrderedDict
import logging

from collections import namedtuple

import copy_reg
import types

__version__ = 0.3

#############################################################################
# Enable pickling of instance methods.
#############################################################################
def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))
copy_reg.pickle(types.MethodType, reduce_method)

#############################################################################
# Setup logging.
#############################################################################
def setup_logger(name):
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.WARNING)
    return logger
logger = setup_logger(__name__)

#############################################################################
# Workflow run function.
#############################################################################

def run(workflow, mapper=map):
    """Run the workflow."""
    if not os.path.isdir(workflow.output_directory):
        os.mkdir(workflow.output_directory)

    if len(workflow.nodes) > 0:
        for node in workflow.nodes:
            run(node, mapper=mapper)
    else:
        try:
            workflow.process()
        except NotImplementedError:
            mapper(workflow.execute, workflow.get_tasks())
        
#############################################################################
# Settings.
#############################################################################

class _NestedClassGetter(object):
    """See also:
    http://stackoverflow.com/questions/1947904/how-can-i-pickle-a-nested-class-in-python/11493777#11493777
    """
    def __call__(self, containing_class, class_name):
        nested_class = getattr(containing_class, class_name)
        nested_instance = _NestedClassGetter()
        nested_instance.__class__ = nested_class
        return nested_instance

class _MetaSettings(type):
    """Meta class for storing node settings."""
    def __init__(cls, name, bases, attrs):
        cls.allowed_attrs = [a for a in attrs.keys()
                             if not a.startswith('__')
                             and not callable(cls.__dict__[a])]

class BaseSettings(object):
    """Base class for storing node settings.

    :raises: RuntimeError if one tries to set a setting that has not been
             specified in the class.
    """
    __metaclass__ = _MetaSettings
    map = map

    def __setattr__(self, name, value):
        if name == 'map':
            self.__dict__[name] = value
        elif name not in self.__class__.allowed_attrs:
            raise RuntimeError
        self.__dict__[name] = value

    def __reduce__(self):
        """See also:
        http://stackoverflow.com/questions/1947904/how-can-i-pickle-a-nested-class-in-python/11493777#11493777
        """
        state = self.__dict__.copy()
        for key, value in self.items():
            state[key] = value
        return (_NestedClassGetter(),
                (_BaseNode, self.__class__.__name__,),
                state,)

    def _keys(self):
        """Return list of sorted setting names."""
        return sorted([key for key in self.__class__.allowed_attrs])

    def items(self):
        """Return settings as a sorted list of key/value pairs."""
        items = []
        for key in self._keys():
            try:
                items.append((key, self.__dict__[key]))
            except KeyError:
                items.append((key, self.__class__.__dict__[key]))
        return items

    def to_json(self, indent=None):
        """Return json representation of settings.

        Ordered alphabetically."""

        ordered_dict = OrderedDict(self.items())
        return json.dumps(ordered_dict, indent=indent)

    def from_json(self, json_str):
        """Configure the settings from a json string."""
        for key, value in json.loads(json_str).items():
            self.__setattr__(key, value)


#############################################################################
# Private base node.
#############################################################################

class _BaseNode(object):
    """Base class for the processing nodes."""
    class Settings(BaseSettings):
        pass

    def __init__(self):
        self.settings = self.__class__.Settings()
        self._output_directory = ''
        self._parent = None
        self.nodes = []
        self.configure()

    def configure(self):
        """Configure a meta node."""
        pass

    @property
    def output_directory(self):
        """Return the node's working directory.

        :returns: path to output directory
        """
        if self._parent is None:
            return os.path.join(self._output_directory,
                                self.__class__.__name__)
        else:
            return os.path.join(self._parent.output_directory,
                                self.__class__.__name__)

    @output_directory.setter
    def output_directory(self, directory):
        """Set the worflow's working directory.

        :param directory: directory where the workflow will create sub-directories
        :raises: RuntimeError if called on a node that is not the top-level one
        """
        if self._parent is None:
            self._output_directory = directory
        else:
            raise RuntimeError('Working directory cannot be set on a sub node.')
        
    def process(self):
        """Process the node.

        Override this function to implement custom processing logic.

        This function does not get called if the execute and get_task
        functions have been implemented.

        This function is useful when it is difficult to set up a function and
        input for a map(funciton, input) logic. For example when going from
        many files to one.
        """
        raise NotImplementedError
        
    def get_tasks(self):
        """Return a list of task.

        Or rather a list of tuples of inputs for each task.
        
        Override this function to implement the desired input for the execution logic.
        The execute command is called by the process fuction using map:

        map(self.execute, self.get_tasks)
        """
        raise NotImplementedError

    def execute(self, task_input):
        """Execute a single task.
        
        Override this function to implement the desired execution logic.
        The execute command is called by the process fuction using map:

        map(self.execute, self.get_tasks)
        """
        raise NotImplementedError

    def add_node(self, node):
        """Add a node to the meta node.

        :param node: node to be added to the meta node
        :returns: the added node
        """
        node._parent = self
        self.nodes.append(node)
        return node

#############################################################################
# Input and output
#############################################################################

class FilePath(str):
    """Class for dealing with file paths.

    Subclass of ``str``.
    """

    @property
    def exists(self):
        """Wether or not the file exists."""
        return os.path.isfile(self)

    def is_more_recent_than(self, other):
        """Wether or not the file is more recent than the other file."""
        return os.path.getmtime(self) > os.path.getmtime(other)

class _InOne(object):
    """Base class for nodes that take one input."""

    def __init__(self, input_obj):
        self.input_obj = input_obj

    @property
    def input_file(self):
        """Return the input file name.
        
        :returns: class:`workflow.FilePath`
        """
        return FilePath(self.input_obj)

class _InMany(object):
    """Base class for nodes that take many inputs."""

    def __init__(self, input_obj):
        self.input_obj = input_obj

    @property
    def input_files(self):
        """Return list containing input file names / tuples of file names.
        
        If the input_obj was a path or a node this function yields filenames.
        If the input_obj was a tuple or list of paths/nodes this function
        yields a tuple of filenames.

        :returns: list of :class:`workflow.FilePath` instances or list of
                  tuples of :class:`workflow.FilePath` instances
        """

        def yield_files(input_obj):
            """Recursive function for yielding files."""
            if isinstance(input_obj, _OutMany):
                # If the input object is an instance of _OutMany it will have
                # access to the output_files property.
                for fname in input_obj.output_files:
                    yield FilePath(fname)
            elif hasattr(input_obj, '__iter__'):
                # This comes after isinstance(input_obj, _OutMany) because some
                # unit test make use of MagicMock that has an "__iter__"
                # attribute.

                # The input object is a tuple or list of input objects.
                all_files = []
                for iobj in input_obj:
                    all_files.append(yield_files(iobj))
                for fnames in zip(*all_files):
                    yield fnames
            else:
                # At this point we assume that we have been given a path to an
                # input directory.
                for fname in os.listdir(input_obj):
                    yield FilePath(os.path.join(input_obj, fname))

        return [f for f in yield_files(self.input_obj)]

class _OutMany(object):
    """Base class for nodes that return many outputs."""

    @property
    def output_files(self):
        """Return list of output file names.
        
        :returns: list of :class:`workflow.FilePath` instances
        """
        return [FilePath(os.path.join(self.output_directory, fname))
                for fname in os.listdir(self.output_directory)]

    def get_output_file(self, fname, enumerator=None):
        """Returns output file name.

        This is a helper function to create meaningful output filenames.
        
        :param fname: input file name
        :param enumerator: unique id (useful if the input file names are not
                           unique)
        :returns: :class:`workflow.FilePath`
        """
        logger.info('fname: {}'.format(fname))
        fname = os.path.basename(fname)
        if enumerator is not None:
            name, suffix = fname.split('.')
            fname = '{}_{}.{}'.format(name, enumerator, suffix)
        return FilePath(os.path.join(self.output_directory, fname))
        
class _OutOne(object):
    """Base class for nodes that produce one output."""

    def __init__(self, output_obj):
        self.output_obj = output_obj

    @property
    def output_file(self):
        """Return the output file name.
        
        :returns: :class:`workflow.FilePath`
        """
        return FilePath(self.output_obj)


#############################################################################
# Public nodes.
#############################################################################

Task = namedtuple('Task', ['input_file', 'output_file', 'settings'])

class OneToManyNode(_BaseNode, _InOne, _OutMany):
    """One to many processing node."""
    def __init__(self, input_obj):
        _InOne.__init__(self, input_obj)
        # Run base code initialisation after in case _BaseNode.configure tries
        # to access input_obj added by _InOne.__init__.
        _BaseNode.__init__(self)

class ManyToManyNode(_BaseNode, _InMany, _OutMany):
    """Many to many processing node."""
    def __init__(self, input_obj):
        _InMany.__init__(self, input_obj)
        # Run base code initialisation after in case _BaseNode.configure tries
        # to access input_obj added by _InMany.__init__.
        _BaseNode.__init__(self)

    def get_tasks(self):
        """Return list of named tuples of input values for execute.

        :returns: list of Task(input_file, output_file, settings)
        """

        tasks = []
        for input_fn in self.input_files:
            output_fn = self.get_output_file(input_fn)
            if output_fn.exists and output_fn.is_more_recent_than(input_fn):
                continue
            tasks.append(Task(input_fn, self.get_output_file(input_fn), self.settings))
        return tasks

class ManyToOneNode(_BaseNode, _InMany, _OutOne):
    """Many to one processing node."""
    def __init__(self, input_obj, output_obj):
        _InMany.__init__(self, input_obj)
        _OutOne.__init__(self, output_obj)
        # Run base code initialisation after in case _BaseNode.configure tries
        # to access input_obj/ouput_obj added by _InMany/_OutOne.__init__.
        _BaseNode.__init__(self)

class OneToOneNode(_BaseNode, _InOne, _OutOne):
    """One to one node."""
    def __init__(self, input_obj, output_obj):
        _InOne.__init__(self, input_obj)
        _OutOne.__init__(self, output_obj)
        # Run base code initialisation after in case _BaseNode.configure tries
        # to access input_obj/ouput_obj added by _InOne/_OutOne.__init__.
        _BaseNode.__init__(self)
