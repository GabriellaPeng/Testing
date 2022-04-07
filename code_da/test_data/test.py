import math

def same_digit(num):
    for i in range(1, 7):
        str_num = str(num)
        str_num2 = str(num * i)
        if sorted(str_num) != sorted(str_num2):
            return False
    return True

for num in range(1, 10000000):
    if same_digit(num):
        for i in range(1, 7):
            print(num*i)


def get_prime_factor(num):
    output = [ ]
    for i in range(2, math.ceil(num/2)):
        if num%i != 0:
            return [num]
        elif num%i == 0:
            output.append(i)
    return output