"""Test code samples for comment generation testing."""

# Simple function - addition
def add(a, b):
    return a + b


# Function with type hints
def calculate_area(length, width):
    return length * width


# Function with conditional logic
def is_even(number):
    if number % 2 == 0:
        return True
    return False


# Function with list processing
def filter_positive(numbers):
    return [n for n in numbers if n > 0]


# Function with string manipulation
def reverse_string(text):
    return text[::-1]


# Function with default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"


# Simple class with methods
class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self.value


# Class with properties
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


# Class with inheritance
class Animal:
    def __init__(self, name):
        self.name = name


class Dog(Animal):
    def __init__(self, name):
        super().__init__(name)
