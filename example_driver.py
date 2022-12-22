from autodiff import AD, ForwardMode, ReverseMode

# initialize the function f
f = lambda x: 2 * AD.sin(x) + 10

# alternatively, the user can initalize the function f as a regular python function
def f(x):
    return 2 * AD.sin(x) + 10

# pass it as an argument to initialize ForwardMode
fm = ForwardMode(f, ['x'])

# initialize x as a list
x = [1]

# print f(x), where x = 1
print(fm.get_f(x)) # output: 11.682941969615793

# print f'(x), where x = 1
print(fm.get_f_prime(x)) # output: 1.0806046117362795

# pass the function as an argument to initialize ReverseMode
rm = ReverseMode(f, ['x'])

# print f(x), where x = 1
print(rm.get_f(x)) # output: 11.682941969615793

# print f'(x), where x = 1
print(rm.get_f_prime(x)) # output: 1.0806046117362795