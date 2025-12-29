from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool


class AddInput(BaseModel):
    numbers: List[float] = Field(..., description="Numbers to add")

@tool(name_or_callable="add_numbers",args_schema=AddInput)
def add_numbers(numbers: List[float]) -> float:
    """
    Calculates the sum of a list of numbers.

    Args:
        numbers: A list of floating-point numbers.

    Returns:
        The sum of all numbers in the list.
    """
    return sum(numbers)

@tool(parse_docstring=True)
def power_calc(x: float, exp: int) -> float:
    """
    Calculates the power of a number.

    Args:
        x: The base number (can be an integer or a float).
        exp: The exponent (an integer).

    Returns:
        The result of x raised to the power of exp.
    """
    return pow(x, exp)


@tool(name_or_callable="calc_factorial", parse_docstring=True)
def factorio(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.

    Raises:
        ValueError: If n is a negative integer.
    """

    def factorial(n: int) -> int:
    
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        elif n == 0:
            return 1
        else:
            return n * factorial(n-1)
    
    return factorial(n)