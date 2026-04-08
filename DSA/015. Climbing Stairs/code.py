# Climbing Stairs
# Description: Count distinct ways to climb stairs taking 1 or 2 steps.
# Example: n = 3 → 3
# Constraints: Dynamic programming

def climbStairs(n):
    if n <= 2:
        return n

    prev1 = 2  # ways to reach step 2
    prev2 = 1  # ways to reach step 1

    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1


# Example usage
if __name__ == "__main__":
    n = 3
    print(climbStairs(n))  # Output: 3
