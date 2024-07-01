'''
这段代码定义了一些用于解析和反解析不同数据类型的函数，主要是针对列表、集合（FrozenSet 和 Set），以及位表示（用布尔值列表表示）的数据类型。

float_parse_entry 和 float_unparse_entry:

float_parse_entry 函数接受一个字符串 line，该字符串包含空格分隔的浮点数，将其解析为浮点数列表。
float_unparse_entry 函数接受一个浮点数列表 entry，将其转换为字符串，浮点数之间用空格分隔。
int_parse_entry 和 int_unparse_entry:

int_parse_entry 函数接受一个字符串 line，该字符串包含空格分隔的整数，将其解析为整数的冻结集合（FrozenSet[int]）。
int_unparse_entry 函数接受一个整数集合（Set[int] 或 FrozenSet[int]），将其转换为字符串，整数之间用空格分隔。
bit_parse_entry 和 bit_unparse_entry:

bit_parse_entry 函数接受一个字符串 line，该字符串包含的字符被解释为布尔值，将其解析为布尔值列表。
bit_unparse_entry 函数接受一个布尔值列表 entry，将其转换为字符串，布尔值之间用空格分隔，表示位的值为 1 或 0。
'''

from typing import List, Union, Set, FrozenSet


def float_parse_entry(line: str) -> List[float]:
    return [float(x) for x in line.strip().split()]


def float_unparse_entry(entry: List[float]) -> str:
    return " ".join(map(str, entry))


def int_parse_entry(line: str) -> FrozenSet[int]:
    return frozenset([int(x) for x in line.strip().split()])


def int_unparse_entry(entry: Union[Set[int], FrozenSet[int]]) -> str:
    return " ".join(map(str, map(int, entry)))


def bit_parse_entry(line: str) -> List[bool]:
    return [bool(int(x)) for x in list(line.strip().replace(" ", "").replace("\t", ""))]


def bit_unparse_entry(entry: List[bool]) -> str:
    return " ".join(map(lambda el: "1" if el else "0", entry))