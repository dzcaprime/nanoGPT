import json
import re


def grade_exact(output, expected):
    return output == expected


def grade_contains(output, expected):
    return str(expected) in output


def grade_json(output, expected=None):
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return False
    if expected is None:
        return True
    return value == expected


def grade_number(output, expected, tolerance=0.0):
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", output)
    if not matches:
        return False
    return abs(float(matches[-1]) - float(expected)) <= float(tolerance)


def grade_example(example, output):
    grader = example["grader"]
    expected = example.get("expected")
    if grader == "exact":
        passed = grade_exact(output, expected)
    elif grader == "contains":
        passed = grade_contains(output, expected)
    elif grader == "json":
        passed = grade_json(output, expected)
    elif grader == "number":
        passed = grade_number(output, expected, example.get("tolerance", 0.0))
    else:
        raise ValueError(f"unknown grader: {grader}")
    return {"passed": passed, "score": 1.0 if passed else 0.0, "reason": "passed" if passed else "failed"}
