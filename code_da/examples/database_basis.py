import numpy as np

# binary search complexity: LogN
def binary_search(arr, target):
    left, right= 0, len(arr)-1
    while left <= right:
        mid  = right//2
        if arr[mid] == target:
            return mid
        elif target < arr[mid] :
            right = mid -1
        else:
            left = mid + 1
    return -1



