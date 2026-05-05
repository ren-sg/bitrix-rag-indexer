from dataclasses import dataclass


@dataclass(frozen=True)
class PhpSymbol:
    kind: str
    name: str
    line: int

@dataclass(frozen=True)
class PhpContext:
    namespace: str | None
    uses: list[str]
    symbols: list[PhpSymbol]

@dataclass(frozen=True)
class PhpDocConfig:
    enabled: bool = True
    include_description: bool = True
    include_tags: tuple[str, ...] = ("deprecated",)
    max_chars: int = 1200

@dataclass(frozen=True)
class PhpDocInfo:
    description: str
    tags: dict[str, list[str]]

@dataclass(frozen=True)
class PhpPrefixConfig:
    include_uses: bool = True
    include_component_context: bool = True
    include_symbol_fqn: bool = True
    include_symbol_modifiers: bool = True

@dataclass(frozen=True)
class PhpPayloadConfig:
    include_uses: bool = True
