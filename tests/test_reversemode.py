# File       : test_reversemode.py
# Description: Test cases for testing the initialization of an automatic
#              differentiation class.

import pytest
import numpy as np

# import names to test
from autodiff.ad import AD
from autodiff.reversemode import ReverseMode

class TestReverseMode():
    """Test class for reverse mode class"""

    ### Test with correct inputs ###
    def test_init_correct(self):
        # Test that the reverse mode class is initialized correctly.
        f1 = lambda x: 2 * AD.sin(x) + 10
        f2 = lambda x: AD.sqrt(AD.exp(x))
        f3 = lambda y: y ** 2 + AD.sinh(y)
        f4 = lambda x, y: 5 ** AD(y)
        f5 = lambda x, y: AD.arctan(x) + 10 * y
        f6 = lambda x, y, z: z + AD.sin(x) / AD.exp(y)

        # Single function, single input
        rm = ReverseMode(f1, ["x"])
        assert rm.f == f1
        assert rm.inputs == list(["x"])
        assert rm.jacobian == False

        # Single function, multiple inputs = arguments
        rm = ReverseMode(f4, ["x", "y"])
        assert rm.f == f4
        assert rm.inputs == list(["x", "y"])
        assert rm.jacobian == False

        # Single function, multiple inputs > arguments
        rm = ReverseMode(f1, ["x", "y"])
        assert rm.f == f1
        assert rm.inputs == list(["x", "y"])
        assert rm.jacobian == False

        # Multiple functions, single input
        rm = ReverseMode([f1, f2], ["x"])
        assert rm.f == list([f1, f2])
        assert rm.inputs == list(["x"])
        assert rm.jacobian == True

        # Multiple functions, single input, same arguments
        rm = ReverseMode([f1, f2], ["x"])
        assert rm.f == list([f1, f2])
        assert rm.inputs == list(["x"])
        assert rm.jacobian == True

        # Multiple functions, single input, different arguments
        rm = ReverseMode([f1, f3], ["x", "y"])
        assert rm.f == list([f1, f3])
        assert rm.inputs == list(["x", "y"])
        assert rm.jacobian == True

        # Multiple functions, multiple inputs, same arguments
        rm = ReverseMode([f4, f5], ["x", "y"])
        assert rm.f == list([f4, f5])
        assert rm.inputs == list(["x", "y"])
        assert rm.jacobian == True

        # Multiple functions, multiple inputs, different arguments
        rm = ReverseMode([f4, f5, f6], ["x", "y", "z"])
        assert rm.f == list([f4, f5, f6])
        assert rm.inputs == list(["x", "y", "z"])
        assert rm.jacobian == True

    ### Tests with wrong inputs ###
    def test_init_wrong(self):
        # Test that the reverse mode class will
        # raise errors if initialized incorrectly.
        
        f1 = lambda x: 2 * AD.sin(x) + 10
        f2 = lambda x: AD.sqrt(AD.exp(x))
        f = 0
        
        # Functions
        # One function: user did not pass in a function
        with pytest.raises(TypeError):
            rm = ReverseMode(0, ["x"])

        # Multiple functions: user did not pass in any functions
        with pytest.raises(TypeError):
            rm = ReverseMode([f, f, f], ["x"])

        # Multiple functions: user did not pass in all functions
        with pytest.raises(TypeError):
            rm = ReverseMode([f1, f2, f], ["x"])

        # Inputs
        # One input: user did not pass in a list
        with pytest.raises(TypeError):
            rm = ReverseMode([f1, f2], "x")

        # One input: user passed in an empty list
        with pytest.raises(ValueError):
            rm = ReverseMode([f1, f2], [])

        # One input: user did not pass in a list-type of strings
        with pytest.raises(TypeError):
            rm = ReverseMode([f1, f2], [1])

        # Multiple inputs: user did not pass in a list-type of strings
        with pytest.raises(TypeError):
            rm = ReverseMode([f1, f2], [1, 2, "x"])

    ### Test with correct inputs ### 
    def test_get_values_correct(self):
        # Test that the reverse mode class is able to return the correct values.
        # of f(x) and f'(x) given x.
        f1 = lambda x: 2 * AD.sin(x) + 10
        f2 = lambda x: AD.sqrt(AD.exp(x))
        f3 = lambda y: y ** 2 + AD.sinh(y)
        f4 = lambda x, y: 5 ** AD.cos(y) + x
        f5 = lambda x, y: AD.arctan(x) + 10 * y
        f6 = lambda x, y, z: z + AD.sin(x) / (2 * AD.exp(y))

        # Single function, single input
        rm = ReverseMode(f1, ["x"])
        assert np.all(rm.get_f([1]) == np.array([2 * np.sin(1) + 10]))
        assert np.all(rm.get_f_prime([1]) == np.array([2 * np.cos(1)]))

        # Single function, multiple inputs = arguments
        rm = ReverseMode(f4, ["x", "y"])
        assert np.all(rm.get_f([1, 2]) == np.array([5 ** np.cos(2) + 1]))
        assert np.all(rm.get_f_prime([1, 2]) == np.array([1, -np.log(5) * 5 ** np.cos(2) * np.sin(2)]))

        # Multiple functions, single input, same arguments
        rm = ReverseMode([f1, f2], ["x"])
        assert np.all(rm.get_f([1]) == np.array([2 * np.sin(1) + 10, np.sqrt(np.exp(1))], dtype = object))
        assert np.all(rm.get_f_prime([1])[0][0] == 2 * np.cos(1))
        assert np.all(np.round(rm.get_f_prime([1])[1][0], 6) == np.round(np.exp(1/2) / 2, 6))

        # Multiple functions, single input, different arguments
        rm = ReverseMode([f1, f3], ["x", "y"])
        print(rm.get_results([1,2]))
        assert np.all(rm.get_f([1, 2]) == np.array([2 * np.sin(1) + 10, 2 ** 2 + np.sinh(2)], dtype = object))
        assert np.all(rm.get_f_prime([1, 2])[0] == np.array([np.array([2 * np.cos(1), 0]), np.array([0, 2 * 2 + np.cosh(2)])], dtype = object)[0])

        # Multiple functions, multiple inputs, same arguments
        rm = ReverseMode([f4, f5], ["x", "y"])
        assert np.all(rm.get_f([1, 2]) == np.array([5 ** np.cos(2) + 1, np.arctan(1) + 10 * 2], dtype = object))
        assert np.all(rm.get_f_prime([1, 2])[0][0] == 1)
        assert np.all(rm.get_f_prime([1, 2])[0][1] == -np.log(5) * 5 ** np.cos(2) * np.sin(2))
        assert np.all(rm.get_f_prime([1, 2])[1][0] == 1/2)
        assert np.all(rm.get_f_prime([1, 2])[1][1] == 10)

    ### Test with incorrect inputs ###
    def test_get_values_incorrect(self):
        # Test that the reverse mode class is able raise
        # errors if incorrect inputs are being used.
        f1 = lambda x: 2 * AD.sin(x) + 10
        f2 = lambda x: AD.sqrt(AD.exp(x))
        f3 = lambda y: y ** 2 + AD.sinh(y)
        f4 = lambda x, y: 5 ** AD.cos(y) + x
        f5 = lambda x, y: AD.arctan(x) + 10 * y
        f6 = lambda x, y, z: z + AD.sin(x) / (2 * AD.exp(y))

        rm1 = ReverseMode(f1, ["x"])

        # User input is not of supported type
        with pytest.raises(TypeError):
            rm1.get_results(1)

        # User input is not 1-dimensional
        with pytest.raises(ValueError):
            rm1.get_results([[1]])

        # Function argument not in input
        with pytest.raises(ValueError):
            rm2 = ReverseMode(f4, ["x", "z"])

        # User input is not a list of strings
        with pytest.raises(TypeError):
            rm1 = ReverseMode(f1, [1])

        # User did not pass in inputs
        with pytest.raises(TypeError):
            rm1.get_results()

        # User passed in an empty list
        with pytest.raises(TypeError):
            rm1.get_results([])
