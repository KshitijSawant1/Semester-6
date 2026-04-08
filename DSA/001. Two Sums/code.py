def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        required = target - num
        if required in seen:
            return [seen[required], i]
        seen[num] = i


if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 9

    result = twoSum(nums, target)
    print("Indices:", result)
