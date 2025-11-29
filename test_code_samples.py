def add(a, b):
    return a + b


def is_even(n):
    return n % 2 == 0


def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"


def square(x):
    return x * x


def reverse_string(text):
    return text[::-1]


def filter_positive(nums):
    return [n for n in nums if n > 0]


def calculate_statistics(values):
    total = sum(values)
    count = len(values)
    avg = total / count if count > 0 else 0
    return {"total": total, "average": avg, "count": count}


def sanitize_username(name):
    cleaned = name.strip().lower()
    return "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")


def merge_dicts(a, b):
    merged = a.copy()
    merged.update(b)
    return merged


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


def process_records(records):
    valid = [r for r in records if r.get("active")]
    sorted_records = sorted(valid, key=lambda r: r["timestamp"])
    return {r["id"]: r for r in sorted_records}



class OrderManager:
    def __init__(self):
        self.orders = []

    def add_order(self, order):
        if order not in self.orders:
            self.orders.append(order)

    def export_ids(self):
        return [o["id"] for o in self.orders if "id" in o]

