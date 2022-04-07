# Q1- Two Sum, EASY
def twoSum(nums, target):
    for i, v1 in enumerate(nums):
        for j, v2 in enumerate(nums[i + 1:]):
            if v1 + v2 == target:
                return [i, j + i + 1]


# Q2 Add Two Numbers, Medium
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def addTwoNumbers(self, l1, l2):
        carry = 0
        dummy = ListNode(0)
        p = dummy

        while l1 and l2:
            p.next = ListNode((l1.val + l2.val + carry) % 10)
            carry = (l1.val + l2.val + carry) // 10
            l1 = l1.next
            l2 = l2.next
            p = p.next
        if l1:
            while l1:
                p.next = ListNode((l1.val + carry) % 10)
                carry = (l1.val + carry) // 10
                l1 = l1.next
                p = p.next
        if l2:
            while l2:
                p.next = ListNode((l2.val + carry) % 10)
                carry = (l2.val + carry) // 10
                l2 = l2.next
                p = p.next

        if carry == 1:
            p.next = ListNode(1)

        return dummy.next


# Q21. Merge Two Sorted Lists ?? EASY
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    p = dummy = ListNode(0)
    while l1 and l2:
        if l1.val < l2.val:
            p.next = l1
            l1 = l1.next
        else:
            p.next = l2
            l2 = l2.next
        p = p.next
    p.next = l1 or l2

    return dummy.next

    # why while? why p.next=l1 not p.next=l1.val


# Q53. Maximum Subarray EASY
def maxSubArray(nums):
    if max(nums) < 0:
        return max(nums)
    local_max, global_max = 0, 0
    for i in nums:
        local_max = max(0, local_max + i)
        global_max = max(local_max, global_max)
    return global_max


# Q53 (Q53 upgrade) maximum sum of a subarray with continuous 3 numbers
def maxSubArray_limit(nums, k=3):
    n = len(nums)
    if n < k:
        return None
    global_max = 0
    for i, v in enumerate(nums):
        local_max = sum(nums[i:i + k])
        global_max = max(local_max, global_max)
    return global_max


# Q20 Valid Parentheses EASY
def isValid(self, s: str) -> bool:
    lookup = {"[": "]", "{": "}", "(": ")"}
    stack = []
    for p in s:
        if p in lookup:
            stack.append(p)
        elif len(stack) == 0 or lookup[stack.pop()] != p:
            return False

    return len(stack) == 0


# Q70. Climbing Stairs ?怎么用递归解决？ EASY
def climbStairs(self, n: int) -> int:
    # 可以用递归来解决，Fibonacci sequence
    prev, curr = 0, 1
    for i in range(n):
        prev, curr = curr, prev + curr
    return curr



def findLastDigit(n):
    def fib(f, n):
        # 0th and 1st number of
        # the series are 0 and 1
        f[0] = 0
        f[1] = 1

        # Add the previous 2 numbers
        # in the series and store
        # last digit of result
        for i in range(2, n + 1):
            f[i] = (f[i - 1] + f[i - 2]) % 10;

        return f

    f = [0] * 61

    # Precomputing units digit of
    # first 60 Fibonacci numbers
    f = fib(f, 60)

    return f[n % 60]


# Q101 . Symmetric Tree EASY
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def isSymmetric(self, root):
        if root is None:
            return True

        return self.isSymmetricRecu(root.left, root.right)

    def isSymmetricRecu(self, left, right):
        if left is None and right is None:  # 退路条件
            return True
        if left is None or right is None or left.val != right.val:  # 不符合的条件
            return False
        return self.isSymmetricRecu(left.left, right.right) and self.isSymmetricRecu(
            left.right, right.left)

    # Q104 Maximum Depth of Binary Tree//Class TreeNode
    def maxDepth(self, root):
        if root is None:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1


# Q121 Best Time to Buy and Sell Stock EASY
def maxProfit(prices):
    max_profit, min_price = 0, float('inf')

    for p in prices:
        min_price = min(min_price, p)
        max_profit = max(max_profit, p - min_price)
    return max_profit


# Q136. Single Number
def singleNumber(self, nums):
    r = 0
    for num in nums:
        r ^= num
    return r


# Q141. Linked List Cycle
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def hasCycle(head):
        slow, fast = head, head

        while fast and fast.next:  # ??
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    # Q160 Intersection of Two Linked Lists
    def getIntersectionNode(self, headA, headB):
        p1 = headA
        p2 = headB
        while p1 != p2:
            if not p1:
                p1 = headB
            else:
                p1 = p1.next
            if not p2:
                p2 = headA
            else:
                p2 = p2.next
        return p1


# 169. Majority Element
def majorityElement(nums):
    # mydict = {i: nums.count(i) for i in nums}
    # max_occur_v = max(mydict.values())
    #
    # for k, v in mydict.items():
    #     if v == max_occur_v:
    #         return k
    check = {}

    for number in nums:
        if number in check:
            check[number] += 1
        else:
            check[number] = 1

    for key in check:
        if check[key] > len(nums) / 2:
            return key
    return None
