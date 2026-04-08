# Experiment-03
# PS-V - Continuous Integration Demonstration

def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def run_tests():
    # Error Fixed
    assert add(5, 3) == 8, "Addition Test Failed"
    assert subtract(10, 4) == 6, "Subtraction Test Failed"
    print("All test cases passed successfully.")


if __name__ == "__main__":
    print("Running Experiment-03 Script")
    run_tests()
