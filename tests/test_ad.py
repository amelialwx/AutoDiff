# File       : test_ad.py
# Description: Test cases for testing the initialization of an automatic
#              differentiation class.

import pytest

# import names to test
from autodiff.dual import Dual
from autodiff.ad import AD

class TestAD():
    """Test class for automatic differentiation class"""

    
    ### Tests with correct inputs ###
    def test_init_correct(self):
        # Test that the automatic differentiation class is
        # initialized correctly.
        f1 = lambda x: 2 * AD.sin(x) + 10
        f2 = lambda x: AD.sqrt(AD.exp(x))
        f3 = lambda y: y ** 2 + AD.sinh(y)
        f4 = lambda x, y: 5 ** AD.cos(y) + x
        f5 = lambda x, y: AD.arctan(x) + 10 * y
        f6 = lambda x, y, z: z + AD.sin(x) / AD.exp(y)

        # Single function, single input
        ad = AD(f1, ["x"])
        assert ad.f == f1
        assert ad.inputs == list(["x"])
        assert ad.jacobian == False

        # Single function, multiple inputs = arguments
        ad = AD(f4, ["x", "y"])
        assert ad.f == f4
        assert ad.inputs == list(["x", "y"])
        assert ad.jacobian == False

        # Single function, multiple inputs > arguments
        ad = AD(f1, ["x", "y"])
        assert ad.f == f1
        assert ad.inputs == list(["x", "y"])
        assert ad.jacobian == False

        # Multiple functions, single input
        ad = AD([f1, f2], ["x"])
        assert ad.f == list([f1, f2])
        assert ad.inputs == list(["x"])
        assert ad.jacobian == True

        # Multiple functions, single input, same arguments
        ad = AD([f1, f2], ["x"])
        assert ad.f == list([f1, f2])
        assert ad.inputs == list(["x"])
        assert ad.jacobian == True

        # Multiple functions, single input, different arguments
        ad = AD([f1, f3], ["x", "y"])
        assert ad.f == list([f1, f3])
        assert ad.inputs == list(["x", "y"])
        assert ad.jacobian == True

        # Multiple functions, multiple inputs, same arguments
        ad = AD([f4, f5], ["x", "y"])
        assert ad.f == list([f4, f5])
        assert ad.inputs == list(["x", "y"])
        assert ad.jacobian == True

        # Multiple functions, multiple inputs, different arguments
        ad = AD([f4, f5, f6], ["x", "y", "z"])
        assert ad.f == list([f4, f5, f6])
        assert ad.inputs == list(["x", "y", "z"])
        assert ad.jacobian == True

    ### Tests with wrong inputs ###
    def test_init_wrong(self):
        # Test that the automatic differentiation class will
        # raise errors if initialized incorrectly
        f1 = lambda x: 2 * Dual.sin(x) + 10
        f2 = lambda x: Dual.sqrt(Dual.exp(x))
        f = 0
        
        # Functions
        # One function: user did not pass in a function
        with pytest.raises(TypeError):
            ad = AD(0, ["x"])

        # Multiple functions: user did not pass in any functions
        with pytest.raises(TypeError):
            ad = AD([f, f, f], ["x"])

        # Multiple functions: user did not pass in all functions
        with pytest.raises(TypeError):
            ad = AD([f1, f2, f], ["x"])

        # One function: not all arguments are present in the input
        with pytest.raises(ValueError):
            ad = AD(f1, ["z"])

        # Multiple functions: not all arguments are present in the input
        with pytest.raises(ValueError):
            ad = AD([f1, f2], ["z"])

        # Inputs
        # One input: user did not pass in a list
        with pytest.raises(TypeError):
            ad = AD([f1, f2], "x")

        # One input: user passed in an empty list
        with pytest.raises(ValueError):
            ad = AD([f1, f2], [])

        # One input: user did not pass in a list-type of strings
        with pytest.raises(TypeError):
            ad = AD([f1, f2], [1])

        # Multiple inputs: user did not pass in a list-type of strings
        with pytest.raises(TypeError):
            ad = AD([f1, f2], [1, 2, "x"])

    def test_get_function(self):
        # Test that the automatic differentiation get_function
        # function returns the correct function.
        f1 = lambda x: 2 * AD.sin(x) + 10
        f2 = lambda x: AD.sqrt(AD.exp(x))
        
        ad1 = AD(f1, ["x"])
        ad2 = AD([f1, f2], ["x"])
        
        assert ad1.get_function() == f1
        assert ad2.get_function() == [f1, f2]

    
