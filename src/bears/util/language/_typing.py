import functools
import json
import typing
from abc import ABC
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from autoenum import AutoEnum
from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticSchemaGenerationError,
    ValidationError,
    constr,
    create_model,
    model_validator,
    validate_call,
)

from ._function import call_str_to_params, get_fn_spec, is_function, params_to_call_str
from ._string import NeverFailJsonEncoder, String
from ._structs import as_list, as_set, is_list_like
from ._utils import get_default


def type_str(data: Any) -> str:
    if isinstance(data, type):
        if issubclass(data, Parameters):
            out: str = data.class_name
        else:
            out: str = str(data.__name__)
    else:
        out: str = str(type(data))
    ## Crocodile brackets mess up Aim's logging, they are treated as HTML tags.
    out: str = out.replace("<", "").replace(">", "")
    return out


def is_abstract(Class: Type) -> bool:
    return ABC in Class.__bases__


## Ref: https://stackoverflow.com/a/13624858/4900327
class classproperty(property):
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)

    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)

    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))


def safe_validate_arguments(f):
    try:

        @functools.wraps(f)
        @validate_call(
            config={
                ## Allow population of a field by it's original name and alias (if False, only alias is used)
                "populate_by_name": True,
                ## Perform type checking of non-BaseModel types (if False, throws an error)
                "arbitrary_types_allowed": True,
            }
        )
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        return wrapper
    except PydanticSchemaGenerationError as e:
        raise e
    except Exception as e:
        raise ValueError(
            f"Error creating Pydantic v2 model to validate function '{get_fn_spec(f).resolved_name}':"
            f"\nEncountered Exception: {String.format_exception_msg(e)}"
        )


def check_isinstance(
    x: Optional[Any], y: Union[List[Type], Tuple[Type, ...], Type], raise_error: bool = True
):
    if x is None and y is type(None):
        return True
    assert isinstance(y, type) or (isinstance(y, (list, tuple)) and np.all([isinstance(z, type) for z in y]))
    if (isinstance(y, type) and isinstance(x, y)) or (
        isinstance(y, list) and np.any([isinstance(x, z) for z in y])
    ):
        return True
    if raise_error:
        y_str: str = ", ".join([type_str(_y) for _y in as_list(y)])
        raise TypeError(
            f"Input parameter must be of type `{y_str}`; found type `{type_str(x)}` with value:\n{x}"
        )
    return False


def check_isinstance_or_none(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return True
    return check_isinstance(x, y, raise_error=raise_error)


def check_issubclass_or_none(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return True
    return check_issubclass(x, y, raise_error=raise_error)


def check_issubclass(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return False
    assert isinstance(x, type)
    assert isinstance(y, type) or (isinstance(y, list) and np.all([isinstance(z, type) for z in y]))
    if (isinstance(y, type) and issubclass(x, y)) or (
        isinstance(y, list) and np.any([issubclass(x, z) for z in y])
    ):
        return True
    if raise_error:
        raise TypeError(
            f"Input parameter must be a subclass of type {str(y)}; found type {type(x)} with value {x}"
        )
    return False


def get_classvars(cls) -> List[str]:
    return [
        var_name
        for var_name, typing_ in typing.get_type_hints(cls).items()
        if typing_.__origin__ is typing.ClassVar
    ]


def get_classvars_typing(cls) -> Dict[str, Any]:
    return {
        var_name: typing_.__args__[0]
        for var_name, typing_ in typing.get_type_hints(cls).items()
        if typing.get_origin(typing_) is typing.ClassVar
    }


class Registry(ABC):
    """
    A registry for subclasses. When a base class extends Registry, its subclasses will automatically be registered,
     without any code in the base class to do so explicitly.
    This coding trick allows us to maintain the Dependency Inversion Principle, as the base class does not have to
     depend on any subclass implementation; in the base class code, we can instead retrieve the subclass in the registry
     using a key, and then interact with the retrieved subclass using the base class interface methods (which we assume
     the subclass has implemented as per the Liskov Substitution Principle).

    Illustrative example:
        Suppose we have abstract base class AbstractAnimal.
        This is registered as a registry via:
            class AbstractAnimal(Parameters, Registry, ABC):
                pass
        Then, subclasses of AbstractAnimal will be automatically registered:
            class Dog(AbstractAnimal):
                name: str
        Now, we can extract the subclass using the registered keys (of which the class-name is always included):
            AbstractAnimalSubclass = AbstractAnimal.get_subclass('Dog')
            dog = AbstractAnimalSubclass(name='Sparky')

        We can also set additional keys to register the subclass against:
            class AnimalType(AutoEnum):
                CAT = auto()
                DOG = auto()
                BIRD = auto()

            class Dog(AbstractAnimal):
                aliases = [AnimalType.DOG]

            AbstractAnimalSubclass = AbstractAnimal.get_subclass(AnimalType.DOG)
            dog = AbstractAnimalSubclass(name='Sparky')

        Alternately, the registry keys can be set using the _registry_keys() classmethod:
            class Dog(AbstractAnimal):
                @classmethod
                def _registry_keys(cls) -> List[Any]:
                    return [AnimalType.DOG]
    """

    _registry: ClassVar[Dict[Any, Dict[str, Type]]] = {}  ## Dict[key, Dict[classname, Class]
    _registry_base_class: ClassVar[Optional[Type[BaseModel]]] = None
    _classvars_typing_dict: ClassVar[Optional[Dict[str, Any]]] = None
    _classvars_BaseModel: ClassVar[Optional[Type[BaseModel]]] = None
    _allow_multiple_subclasses: ClassVar[bool] = False
    _allow_subclass_override: ClassVar[bool] = False
    _dont_register: ClassVar[bool] = False
    aliases: ClassVar[Tuple[str, ...]] = tuple()

    def __init_subclass__(cls, **kwargs):
        """
        Register any subclass with the base class. A child class is registered as long as it is imported/defined.
        """
        super().__init_subclass__(**kwargs)
        if cls in Registry.__subclasses__():
            ## The current class is a direct subclass of Registry (i.e. it is the base class of the hierarchy).
            cls._registry: Dict[Any, Dict[str, Type]] = {}
            cls._registry_base_class: Type = cls
            cls.__set_classvars_typing()
        else:
            ## The current class is a subclass of a Registry-subclass, and is not abstract; register this.
            if not is_abstract(cls) and not cls._dont_register:
                cls._pre_registration_hook()
                cls.__set_classvars_typing()
                cls.__validate_classvars_BaseModel()
                cls.__register_subclass()

    @classmethod
    def __set_classvars_typing(cls):
        classvars_typing_dict: Dict[str, Any] = {
            var_name: typing_
            for var_name, typing_ in get_classvars_typing(cls).items()
            if not var_name.startswith("_")
        }
        cls._classvars_typing_dict: ClassVar[Dict[str, Any]] = classvars_typing_dict

        fields = {k: (v, None) for k, v in classvars_typing_dict.items()}

        try:
            _classvars_BaseModel: ClassVar[Optional[Type[BaseModel]]] = create_model(
                f"{cls.__name__}_ClassVarsBaseModel",
                __config__=Parameters.model_config,
                **fields,
            )
            _classvars_BaseModel.model_config = {"extra": "ignore"}
            cls._classvars_BaseModel: ClassVar[Optional[Type[BaseModel]]] = _classvars_BaseModel
        except PydanticSchemaGenerationError as e:
            raise PydanticSchemaGenerationError(
                f"Error creating Pydantic v2 model to validate classvars of '{cls.__name__}':\n"
                f"{String.format_exception_msg(e)}\n"
                f"Fields used:\n{String.pretty(fields)}"
            )

    @classmethod
    def __validate_classvars_BaseModel(cls):
        ## Gives the impression of validating ClassVars on concrete subclasses in the hierarchy.
        classvar_values: Dict[str, Any] = {}
        for classvar, type_ in cls._classvars_typing_dict.items():
            if not hasattr(cls, classvar):
                if ABC not in cls.__bases__:
                    ## Any concrete class must have all classvars set with values.
                    raise ValueError(
                        f'You must set a value for class variable "{classvar}" on subclass "{cls.__name__}".\n'
                        f'Custom type-hints might be one reason why "{classvar}" is not recognized. '
                        f'If you have added custom type-hints, please try removing them and set "{classvar}" like so: `{classvar} = <value>`'
                    )
            else:
                classvar_value = getattr(cls, classvar)
                if hasattr(type_, "__origin__"):
                    if (
                        type_.__origin__ == typing.Union
                        and len(type_.__args__) == 2
                        and type(None) in type_.__args__
                    ):
                        ## It is something like Optional[str], Optional[List[str]], etc.
                        args = set(type_.__args__)
                        args.remove(type(None))
                        classvar_type = next(iter(args))
                    else:
                        classvar_type = type_.__origin__
                    if classvar_type in {set, list, tuple} and classvar_value is not None:
                        classvar_value = classvar_type(as_list(classvar_value))
                classvar_values[classvar] = classvar_value
        classvar_values: BaseModel = cls._classvars_BaseModel(**classvar_values)
        for classvar, type_ in cls._classvars_typing_dict.items():
            if not hasattr(cls, classvar):
                if ABC not in cls.__bases__:
                    ## Any concrete class must have all classvars set with values.
                    raise ValueError(
                        f'You must set a value for class variable "{classvar}" on subclass "{cls.__name__}".\n'
                        f'Custom type-hints might be one reason why "{classvar}" is not recognized. '
                        f'If you have added custom type-hints, please try removing them and set "{classvar}" like so: `{classvar} = <value>`'
                    )
            else:
                setattr(cls, classvar, classvar_values.__getattribute__(classvar))

    @classmethod
    def _pre_registration_hook(cls):
        pass

    @classmethod
    def __register_subclass(cls):
        keys_to_register: List[Any] = []
        for key in [str(cls.__name__)] + as_list(cls.aliases) + as_list(cls._registry_keys()):
            if key is None:
                continue
            elif isinstance(key, (str, AutoEnum)):
                ## Case-insensitive matching:
                key: str = String.str_normalize(key)
            elif isinstance(key, tuple):
                key: Tuple = tuple(
                    ## Case-insensitive matching:
                    String.str_normalize(key_part) if isinstance(key_part, (str, AutoEnum)) else key_part
                    for key_part in key
                )
            keys_to_register.append(key)
        cls.__add_to_registry(keys_to_register, cls)

    @classmethod
    @validate_call
    def __add_to_registry(cls, keys_to_register: List[Any], subclass: Type):
        subclass_name: str = subclass.__name__
        for k in as_set(keys_to_register):  ## Drop duplicates
            if k not in cls._registry:
                cls._registry[k] = {subclass_name: subclass}
                continue
            ## Key is in the registry
            registered: Dict[str, Type] = cls._registry[k]
            registered_names: Set[str] = set(registered.keys())
            assert len(registered_names) > 0, f"Invalid state: key '{k}' is registered to an empty dict"
            if subclass_name in registered_names and cls._allow_subclass_override is False:
                raise KeyError(
                    f"A subclass with name '{subclass_name}' is already registered "
                    f"against key '{k}' for registry under '{cls._registry_base_class}'; "
                    f"overriding subclasses is not permitted."
                )
            elif subclass_name not in registered_names and cls._allow_multiple_subclasses is False:
                assert len(registered_names) == 1, (
                    f"Invalid state: _allow_multiple_subclasses is False but we have multiple subclasses registered "
                    f"against key {k}"
                )
                raise KeyError(
                    f"Key {k} already is already registered to subclass {next(iter(registered_names))}; registering "
                    f"multiple subclasses to the same key is not permitted."
                )
            cls._registry[k] = {
                **registered,
                ## Add or override the subclass names
                subclass_name: subclass,
            }

    @classmethod
    def get_subclass(
        cls,
        key: Any,
        raise_error: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Union[Type, List[Type]]]:
        if isinstance(key, (str, AutoEnum)):
            Subclass: Optional[Dict[str, Type]] = cls._registry.get(String.str_normalize(key))
        else:
            Subclass: Optional[Dict[str, Type]] = cls._registry.get(key)
        if Subclass is None:
            if raise_error:
                available_keys: str = "\n".join(sorted(set(cls._registry.keys())))
                raise KeyError(
                    f'Could not find subclass of {cls} using key: "{key}" (type={type(key)}). '
                    f"Available keys are:\n{available_keys}"
                )
            return None
        if len(Subclass) == 1:
            return next(iter(Subclass.values()))
        return list(Subclass.values())

    @classmethod
    def subclasses(cls, keep_abstract: bool = False) -> Set[Type]:
        available_subclasses: Set[Type] = set()
        for k, d in cls._registry.items():
            for subclass in d.values():
                if subclass == cls._registry_base_class:
                    continue
                if is_abstract(subclass) and keep_abstract is False:
                    continue
                if isinstance(subclass, type) and issubclass(subclass, cls):
                    available_subclasses.add(subclass)
        return available_subclasses

    @classmethod
    def remove_subclass(cls, subclass: Union[Type, str]):
        name: str = subclass
        if isinstance(subclass, type):
            name: str = subclass.__name__
        for k, d in cls._registry.items():
            for subclass_name, subclass in list(d.items()):
                if String.str_normalize(subclass_name) == String.str_normalize(name):
                    d.pop(subclass_name, None)

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return None


## Ref: https://stackoverflow.com/q/6760685/4900327, Method 2 base class.
## The metaclass method in the above link did not work well with multiple inheritance.
class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(Singleton, cls).__new__(cls)
        return cls.__instance

    @classproperty
    def instance(cls):
        return cls.__instance


ParametersSubclass = TypeVar("ParametersSubclass", bound="Parameters")


class Parameters(BaseModel, ABC):
    ## Ref on Pydantic + ABC: https://pydantic-docs.helpmanual.io/usage/models/#abstract-base-classes
    ## Needed to work with Registry.alias...this needs to be on a subclass of `BaseModel`.
    aliases: ClassVar[Tuple[str, ...]] = tuple()
    dict_exclude: ClassVar[Tuple[str, ...]] = tuple()

    ## Changes from Pydantic V1 to V2 config schema:
    ## https://docs.pydantic.dev/2.1/blog/pydantic-v2-alpha/#changes-to-config
    model_config = ConfigDict(
        ## Only string literal is needed for extra parameter
        ## https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.extra
        extra="forbid",
        ## Renamed from "allow_mutation":
        ## https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.frozen
        frozen=True,
        ## Underscores-as-private is enabled permanently (see "V1 to V2" URL above):
        # underscore_attrs_are_private=True,
        ## Renamed from validate_all
        ## https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_default
        validate_default=True,
        ## https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.arbitrary_types_allowed
        arbitrary_types_allowed=True,
    )

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except ValidationError as e:
            errors_str = ""
            for error_i, error in enumerate(e.errors()):
                assert isinstance(error, dict)
                error_msg: str = String.prefix_each_line(error.get("msg", ""), prefix="    ").strip()
                errors_str += f"\n[Error#{error_i + 1}] ValidationError in {error['loc']}: {error_msg}"
                errors_str += f"\n[Error#{error_i + 1}] Input: {String.pretty(error['input'])}"
            raise ValueError(
                f"Cannot create Pydantic instance of type '{self.class_name}', "
                f"encountered following validation errors: {errors_str}"
            )

        except Exception as e:
            error_msg: str = String.prefix_each_line(String.format_exception_msg(e), "    ")
            raise ValueError(
                f"Cannot create Pydantic instance of type '{self.class_name}', "
                f"encountered exception:\n{error_msg}"
            )

    @classproperty
    def class_name(cls) -> str:
        return str(cls.__name__)  ## Will return the child class name.

    @classmethod
    def param_names(cls, **kwargs) -> Set[str]:
        return set(cls.model_json_schema(**kwargs).get("properties", {}).keys())

    @classmethod
    def param_default_values(cls, **kwargs) -> Dict:
        properties = cls.model_json_schema(**kwargs).get("properties", {})
        return {param: prop.get("default") for param, prop in properties.items() if "default" in prop}

    @classmethod
    def set_default_param_values(cls, params: Dict):
        ## Apply default values for fields not present in the input
        for field_name, field in cls.model_fields.items():
            if field_name not in params:
                if field.default is not None:
                    params[field_name] = field.default
                elif field.default_factory is not None:
                    params[field_name] = field.default_factory()

    @classmethod
    def _clear_extra_params(cls, params: Dict) -> Dict:
        return {k: v for k, v in params.items() if k in cls.param_names()}

    def model_dump(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        exclude: Set[str] = as_set(get_default(exclude, [])).union(as_set(self.dict_exclude))
        return super().model_dump(exclude=exclude, **kwargs)

    def json(self, *args, encoder: Optional[Any] = None, indent: Optional[int] = None, **kwargs) -> str:
        if encoder is None:
            encoder = functools.partial(json.dumps, cls=NeverFailJsonEncoder, indent=indent)
        return super().model_dump_json(**kwargs)  # drop encoder to keep it minimal

    @classproperty
    def _constructor(cls) -> ParametersSubclass:
        return cls

    def __str__(self) -> str:
        params_str: str = self.json(indent=4)
        out: str = f"{self.class_name} with params:\n{params_str}"
        return out

    @staticmethod
    def _convert_params(Class: Type[BaseModel], d: Union[Type[BaseModel], Dict]):
        if type(d) is Class:
            return d
        if isinstance(d, BaseModel):
            return Class(**d.model_dump(exclude=None))
        if d is None:
            return Class()
        if isinstance(d, dict):
            return Class(**d)
        raise NotImplementedError(f"Cannot convert object of type {type(d)} to {Class.__class__}")

    def update_params(self, **new_params) -> Generic[ParametersSubclass]:
        ## Since Parameters class is immutable, we create a new one:
        overridden_params: Dict = {
            **self.model_dump(exclude=None),
            **new_params,
        }
        return self._constructor(**overridden_params)

    def copy(self, **kwargs) -> Generic[ParametersSubclass]:
        return super().model_copy(**kwargs)

    def clone(self, **kwargs) -> Generic[ParametersSubclass]:
        return self.copy(**kwargs)


class UserEnteredParameters(Parameters):
    """
    Case-insensitive Parameters class.
    Use this for configs classes where you expect to read from user-entered input, which might have any case.
    IMPORTANT: the param names in the subclass must be in LOWERCASE ONLY.
    Ref: https://github.com/samuelcolvin/pydantic/issues/1147#issuecomment-571109376
    """

    @model_validator(mode="before")
    @classmethod
    def convert_params_to_lowercase(cls, values: Dict) -> Dict:
        return {str(k).strip().lower(): v for k, v in values.items()}


class MutableParameters(Parameters):
    model_config = ConfigDict(
        ## replaces allow_mutation=True from Pydantic v1:
        frozen=False,
        ## type-checking when setting parameters.
        ## Ref: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_assignment
        validate_assignment=False,
    )


class MutableUserEnteredParameters(UserEnteredParameters, MutableParameters):
    pass


class MappedParameters(Parameters, ABC):
    """
    Allows creation of a Parameters instance by mapping from a dict.
    From this dict, the 'name' key will be used to look up the cls.mapping_dict dictionary, and retrieve the corresponding
    class. This class will be instantiated using the other values in the dict.
    """

    mapping_dict: ClassVar[Dict[Union[Tuple[str, ...], str], Any]]

    model_config = ConfigDict(
        extra="allow",
    )

    name: constr(min_length=1)
    args: Tuple = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isinstance(cls.mapping_dict, dict) or len(cls.mapping_dict) == 0:
            raise ValueError(
                f"Lookup must be a non-empty dict; found: "
                f"type {type(cls.mapping_dict)} with value: {cls.mapping_dict}"
            )
        for key, val in list(cls.mapping_dict.items()):
            if is_list_like(key):
                for k in key:
                    cls.mapping_dict[String.str_normalize(k)] = val
            else:
                cls.mapping_dict[String.str_normalize(key)] = val

    @model_validator(mode="before")
    @classmethod
    def check_mapped_params(cls, values: Dict) -> Dict:
        if String.str_normalize(values["name"]) not in cls.mapping_dict:
            raise ValueError(
                f'''`name`="{values["name"]}" was not found in the lookup. '''
                f"""Valid values for `name`: {set(cls.mapping_dict.keys())}"""
            )
        return values

    def model_dump(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        params: Dict = super(Parameters, self).model_dump(*args, exclude=exclude, **kwargs)
        if exclude is not None and "name" in exclude:
            params.pop("name", None)
        else:
            params["name"] = self.name
        return params

    def __str__(self) -> str:
        params_str: str = self.json(indent=4)
        out: str = f"{self.class_name} with params:\n{params_str}"
        return out

    @classmethod
    def from_call_str(cls, call_str: str) -> Any:
        args, kwargs = call_str_to_params(call_str)
        return cls(args=args, **kwargs)

    def mapped_callable(self) -> Any:
        return self.mapping_dict[String.str_normalize(self.name)]

    @property
    def kwargs(self) -> Dict:
        return self.model_dump(exclude={"name", "args"} | set(self.dict_exclude))

    def to_call_str(self) -> str:
        args: List = list(self.args)
        kwargs: Dict = self.kwargs
        callable: Callable = self.mapped_callable()
        if is_function(callable) or isinstance(callable, type):
            callable_name: str = callable.__name__
        else:
            callable_name: str = str(callable)
        return params_to_call_str(
            callable_name=callable_name,
            args=args,
            kwargs=kwargs,
        )

    @classmethod
    @safe_validate_arguments
    def of(
        cls,
        name: Optional[Union[Parameters, Dict, str]],
        **params,
    ) -> Optional[Any]:
        if name is None:
            return None
        if isinstance(name, cls):
            return name
        if isinstance(name, dict):
            return cls(**name)
        if isinstance(name, str):
            if "(" in name or ")" in name:
                return cls.from_call_str(name)
            else:
                return cls(**{"name": name, **params})
        raise ValueError(f"Unsupported value for `name`: {name}")

    def initialize(self, **kwargs) -> Any:
        return self.mapped_callable()(*self.args, **self.kwargs, **kwargs)
