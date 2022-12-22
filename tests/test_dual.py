# File       : test_dual.py
# Description: Test cases for testing Dual number elementary operations and
#              trigonometric functions

import pytest
import numpy as np

# import names to test
from autodiff.ad import AD
from autodiff.dual import Dual

class TestDual():
    """Test class for dual numbers"""

    def test_init(self):
        # Test that dual numbers are initialized correctly.
        d = Dual(1, 2)
        assert d.real == 1
        assert d.dual == 2

    def test_addition(self):
        # Test that the implementation only supports addition for mixed
        # operands of type int, float, or Dual.
        # Implementation should throw a TypeError when other types are used
        # for addition.
        a = Dual(1, 2)

        # Dual numbers
        b = Dual(3, 4)
        c = a + b
        assert c.real == 4
        assert c.dual == 6

        # Dual and integer numbers
        c = a + 3
        assert c.real == 4
        assert c.dual == 2

        # Dual and float numbers
        c = a + 3.1
        assert c.real == 4.1
        assert c.dual == 2

        # Dual number and string, should raise an error
        with pytest.raises(TypeError):
            c = a + '3'
        

    def test_subtraction(self):
        # Test that the implementation only supports subtraction for mixed
        # operands of type int, float, or Dual.
        # Implementation should throw a TypeError when other types are used
        # for subtraction.
        a = Dual(1, 2)

        # Dual numbers
        b = Dual(3, 4)
        c = a - b
        assert c.real == -2
        assert c.dual == -2

        # Dual and integer numbers
        c = a - 3
        assert c.real == -2
        assert c.dual == 2

        # Dual and float numbers
        c = a - 3.1
        assert c.real == -2.1
        assert c.dual == 2

        # Dual number and string, should raise an error
        with pytest.raises(TypeError):
            c = a - '3'

    def test_multiplication(self):
        # Test that the implementation only supports multiplication for mixed
        # operands of type int, float, or Dual.
        # Implementation should throw a TypeError when other types are used
        # for multiplication.
        a = Dual(1, 2)

        # Dual numbers
        b = Dual(3, 4)
        c = a * b
        assert c.real == 3
        assert c.dual == 10

        # Dual and integer numbers
        c = a * 3
        assert c.real == 3
        assert c.dual == 6

        # Dual and float numbers
        c = a * 3.1
        assert c.real == 3.1
        assert c.dual == 6.2

        # Dual number and string, should raise an error
        with pytest.raises(TypeError):
            c = a * '3'

    def test_division(self):
        # Test that the implementation only supports division for mixed operands
        # of type int, float or dual.
        # Implementation should throw a TypeError when other types are used for
        # division.
        a = Dual(1, 2)

        # Dual numbers
        b = Dual(3, 4)
        c = a / b
        assert c.real == 1/3
        assert c.dual == -4/9 + 2 * (1/3)

        # Dual and integer numbers
        c = a / 3
        assert c.real == 1/3
        assert c.dual == 2/3

        # Dual and float numbers
        c = a / 3.1
        assert c.real == 1/3.1
        assert c.dual == 2/3.1

        # Dual number and string, should raise an error
        with pytest.raises(TypeError):
            c = a / '3'

    def test_reflective_operators(self):
        # Test reflective operators for the Dual type
        # Reflective operators only supports mixed operands of type int, float
        # or Dual.
        # Implementation should throw a TypeError when other types are used.
        a = Dual(1, 2)

        # ADDITION

        # Dual numbers (reverse)
        b = Dual(3, 4)
        c = b + a
        assert c.real == 4
        assert c.dual == 6

        # Dual and integer numbers (reverse)
        c = 3 + a
        assert c.real == 4
        assert c.dual == 2

        # Dual and float numbers (reverse)
        c = 3.1 + a
        assert c.real == 4.1
        assert c.dual == 2

        # Dual number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' + a

        # SUBTRACTION
        # Dual numbers (reverse)
        b = Dual(3, 4)
        c = b - a
        assert c.real == 2
        assert c.dual == 2

        # Dual and integer numbers (reverse)
        c = 3 - a
        assert c.real == 2
        assert c.dual == -2

        # Dual and float numbers (reverse)
        c = 3.1 - a
        assert c.real == 2.1
        assert c.dual == -2

        # Dual number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' - a

        # MULTIPLICATION
        # Dual numbers (reverse)
        b = Dual(3, 4)
        c = b * a
        assert c.real == 3
        assert c.dual == 10

        # Dual and integer numbers (reverse)
        c = 3 * a
        assert c.real == 3
        assert c.dual == 6

        # Dual and float numbers (reverse)
        c = 3.1 * a
        assert c.real == 3.1
        assert c.dual == 6.2

        # Dual number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' * a

        # DIVISION
        # Dual numbers (reverse)
        b = Dual(3, 4)
        c = b / a
        assert c.real == 3
        assert c.dual == -2

        # Dual and integer numbers (reverse)
        c = 3 / a
        assert c.real == 3
        assert c.dual == -6

        # Dual and float numbers (reverse)
        c = 3.1 / a
        assert c.real == 3.1
        assert c.dual == -6.2

        # Dual number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' / a

    
    def test_neg(self):
        # Test that Dual numbers can be negated.
        a = Dual(1, 2)
        c = -a
        assert c.real == -1
        assert c.dual == -2


    def test_pow(self):
        # Test that the implementation only supports mixed operands of
        # type int, float, or Dual.
        # Implementation should throw a TypeError when other types are
        # used.
        a = Dual(1, 2)

        # Dual numbers
        b = Dual(3, 4)
        c = a ** b
        assert c.real == 1
        assert c.dual == 6

        # Dual number and integers
        c = a ** 3
        assert c.real == 1
        assert c.dual == 6

        # Dual number and floats
        c = a ** 3.1
        assert c.real == 1
        assert c.dual == 6.2

        # Dual number and string, should raise an error
        with pytest.raises(TypeError):
            c = a ** '3'

    def test_rpow(self):
        # This only happens if we are taking an integer or float to
        # the power of a Dual number.
        a = Dual(1, 2)

        # Dual number and integers (reverse)
        c = 3 ** a
        assert c.real == 3
        assert c.dual == 3 * np.log(3) * 2

        # Dual number and float (reverse)
        c = 3.1 ** a
        assert c.real == 3.1
        assert c.dual == 3.1 * np.log(3.1) * 2

        # Dual number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' ** a

    def test_elementary(self):
        # Test elementary functions for the Dual type.
        # arcsin and arccos have specific domains and should raise
        # an error if the Dual number is outside of the domain
        # range.
        a = Dual(1, 2)
        b = Dual(0, 1)

        # sin
        c = AD.sin(a)
        assert c.real == np.sin(1)
        assert c.dual == np.cos(1) * 2

        # cos
        c = AD.cos(a)
        assert c.real == np.cos(1)
        assert c.dual == np.sin(1) * -2

        # tan
        c = AD.tan(a)
        assert c.real == np.tan(1)
        assert c.dual == 2 / np.cos(1) ** 2

        # arcsin
        c = AD.arcsin(b)
        assert c.real == 0
        assert c.dual == 1

        # arcsin with Dual real not within -1 and 1, should raise an error
        with pytest.raises(ValueError):
            c = AD.arcsin(a)

        # arccos
        c = AD.arccos(b)
        assert c.real == np.arccos(0)
        assert c.dual == -1

        # arccos with Dual real not within -1 and 1, should raise an error
        with pytest.raises(ValueError):
            c = AD.arccos(a)

        # arctan
        c = a.arctan()
        assert c.real == np.arctan(1)
        assert c.dual == 1

        # exp
        c = AD.exp(a)
        assert c.real == np.exp(1)
        assert c.dual == np.exp(1) * 2

        # log
        c = AD.log(a, 2)
        assert c.real == 0
        assert c.dual == 1/np.log(2) * 2

        # log with string as base, should raise an error
        with pytest.raises(TypeError):
            c = AD.log(a, "base")

        # log with base lesser than or equal to zero, should raise an error
        with pytest.raises(ValueError):
            c = AD.log(a, 0)

        # log with Dual real lesser than or equal to zero, should raise an error
        with pytest.raises(ValueError):
            c = AD.log(b, 2)

        # sqrt
        c = AD.sqrt(a)
        assert c.real == 1
        assert c.dual == 1
        
        # square root of a dual number with the real part lesser than zero, should raise an error
        d = Dual(-1, 1)
        with pytest.raises(ValueError):
            c = AD.sqrt(d)

        # standard logistic
        c = AD.standard_logistic(a)
        assert c.real == 1 / (1 + np.exp(-1))
        assert c.dual == (np.exp(-1) * -2) * -1 * ((1 + np.exp(-1)) ** (-1 - 1))
        
        # sinh
        c = AD.sinh(a)
        assert c.real == (np.exp(1) - np.exp(-1)) / 2
        assert c.dual == (np.exp(1) * 2 - np.exp(-1) * -2) / 2

        # cosh
        c = AD.cosh(a)
        assert c.real == (np.exp(1) + np.exp(-1)) / 2
        assert c.dual == (np.exp(1) * 2 + np.exp(-1) * -2) / 2

        # tanh
        c = AD.tanh(a)
        assert c.real == (np.exp(1) - np.exp(-1)) / 2 * ((np.exp(1) + np.exp(-1)) / 2) ** -1
        assert c.dual == ((np.exp(1) - np.exp(-1)) / 2) * ((np.exp(1) * 2 + np.exp(-1) * -2) / 2 * -1 * (((np.exp(1) + np.exp(-1)) / 2) ** (-1 -1))) + ((np.exp(1) * 2 - np.exp(-1) * -2) / 2) * (((np.exp(1) + np.exp(-1)) / 2) ** -1)
