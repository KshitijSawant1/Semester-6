# Single Number
# Description: Every element appears twice except one. Find that one.
# Example: [4,1,2,1,2] → 4
# Constraints: Linear time and constant space

def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result


if __name__ == "__main__":
    nums = [4, 1, 2, 1, 2]
    print(singleNumber(nums))  
    
# Output: 4
