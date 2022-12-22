# File       : test_node.py
# Description: Test cases for testing Node elementary operations and
#              trigonometric functions

import pytest
import numpy as np

# import names to test
from autodiff.ad import AD
from autodiff.node import Node

class TestNode():
    """Test class for nodes"""

    def test_init(self):
        # Test that Nodes are initialized correctly.
        d = Node(1)
        assert d.val == 1
        assert d.gradients == ()

    def test_addition(self):
        # Test that the implementation only supports addition for mixed
        # operands of type int, float, or Node.
        # Implementation should throw a TypeError when other types are used
        # for addition.
        a = Node(1)

        # Node numbers
        b = Node(3)
        c = a + b
        assert c.val == 4
        assert c.gradients == ((a, 1), (b, 1))

        # Node and integer numbers
        c = a + 3
        assert c.val == 4
        assert c.gradients == ((a, 1),)

        # Node and float numbers
        c = a + 3.1
        assert c.val == 4.1
        assert c.gradients == ((a, 1),)

        # Node and string, should raise an error
        with pytest.raises(TypeError):
            c = a + '3'
        

    def test_subtraction(self):
        # Test that the implementation only supports subtraction for mixed
        # operands of type int, float, or Node.
        # Implementation should throw a TypeError when other types are used
        # for subtraction.
        a = Node(1)

        # Node 
        b = Node(3)
        c = a - b
        assert c.val == -2
        assert c.gradients == ((a, 1), (b, -1))

        # Node and integer numbers
        c = a - 3
        assert c.val == -2
        assert c.gradients == ((a, 1),)

        # Node and float numbers
        c = a - 3.1
        assert c.val == -2.1
        assert c.gradients == ((a, 1),)

        # Node number and string, should raise an error
        with pytest.raises(TypeError):
            c = a - '3'

    def test_multiplication(self):
        # Test that the implementation only supports multiplication for mixed
        # operands of type int, float, or Node.
        # Implementation should throw a TypeError when other types are used
        # for multiplication.
        a = Node(1)

        # Node
        b = Node(3)
        c = a * b
        assert c.val == 3
        assert c.gradients == ((a, 3), (b, 1))

        # Node and integer numbers
        c = a * 3
        assert c.val == 3
        assert c.gradients == ((a, 3),)

        # Node and float numbers
        c = a * 3.1
        assert c.val == 3.1
        assert c.gradients == ((a, 3.1),)

        # Node number and string, should raise an error
        with pytest.raises(TypeError):
            c = a * '3'

    def test_division(self):
        # Test that the implementation only supports division for mixed operands
        # of type int, float or Node.
        # Implementation should throw a TypeError when other types are used for
        # division.
        a = Node(1)

        # Node
        b = Node(3)
        c = a / b
        assert c.val == 1/3
        print(c.gradients)
        assert c.gradients == ((a, 1/3), (b, -1/3**2))

        # Node and integer numbers
        c = a / 3
        assert c.val == 1/3
        assert c.gradients == ((a, 1/3), )

        # Node and float numbers
        c = a / 3.1
        assert c.val == 1/3.1
        assert c.gradients == ((a, 1/3.1), )

        # Node number and string, should raise an error
        with pytest.raises(TypeError):
            c = a / '3'

    def test_reflective_operators(self):
        # Test reflective operators for the Node type
        # Reflective operators only supports mixed operands of type int, float
        # or Node.
        # Implementation should throw a TypeError when other types are used.
        a = Node(1)

        # ADDITION

        # Node numbers (reverse)
        b = Node(3)
        c = b + a
        assert c.val == 4
        assert c.gradients == ((b, 1), (a, 1))

        # Node and integer numbers (reverse)
        c = 3 + a
        assert c.val == 4
        assert c.gradients == ((a, 1),)

        # Node and float numbers (reverse)
        c = 3.1 + a
        assert c.val == 4.1
        assert c.gradients == ((a, 1),)

        # Node number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' + a

        # SUBTRACTION
        # Node numbers (reverse)
        b = Node(3)
        c = b - a
        assert c.val == 2
        assert c.gradients == ((b, 1), (a, -1))

        # Node and integer numbers (reverse)
        c = 3 - a
        assert c.val == 2
        assert c.gradients == ((a, -1),)

        # Node and float numbers (reverse)
        c = 3.1 - a
        assert c.val == 2.1
        assert c.gradients == ((a, -1),)

        # Node number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' - a

        # MULTIPLICATION
        # Node numbers (reverse)
        b = Node(3, 4)
        c = b * a
        assert c.val == 3
        assert c.gradients == ((b, 1), (a, 3))

        # Node and integer numbers (reverse)
        c = 3 * a
        assert c.val == 3
        assert c.gradients == ((a, 3),)

        # Node and float numbers (reverse)
        c = 3.1 * a
        assert c.val == 3.1
        assert c.gradients == ((a, 3.1),)

        # Node number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' * a

        # DIVISION
        # Node numbers (reverse)
        b = Node(3)
        c = b / a
        assert c.val == 3
        assert c.gradients == ((b, 1),(a, -3))

        # Node and integer numbers (reverse)
        c = 3 / a
        assert c.val == 3
        assert c.gradients == ((a, -3),)

        # Node and float numbers (reverse)
        c = 3.1 / a
        assert c.val == 3.1
        assert c.gradients == ((a, -3.1),)

        # Node number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' / a

    
    def test_neg(self):
        # Test that Node numbers can be negated.
        a = Node(1)
        c = -a
        assert c.val == -1
        assert c.gradients == ((a, -1),)


    def test_pow(self):
        # Test that the implementation only supports mixed operands of
        # type int, float, or Node.
        # Implementation should throw a TypeError when other types are
        # used.
        a = Node(1)

        # Node
        b = Node(3)
        c = a ** b
        assert c.val == 1
        assert c.gradients == ((a, 3), (b, 0))

        # Node number and integers
        c = a ** 3
        assert c.val == 1
        assert c.gradients == ((a, 3),)

        # Node number and floats
        c = a ** 3.1
        assert c.val == 1
        assert c.gradients == ((a, 3.1),)

        # Node number and string, should raise an error
        with pytest.raises(TypeError):
            c = a ** '3'

    def test_rpow(self):
        # This only happens if we are taking an integer or float to
        # the power of a Node number.
        a = Node(1)

        # Node
        b = Node(3)
        c = b ** a
        assert c.val == 3
        assert c.gradients == ((b, 1), (a, 3*np.log(3)))

        # Node number and integers (reverse)
        c = 3 ** a
        assert c.val == 3
        assert c.gradients == ((a, 3 * np.log(3)),)

        # Node number and float (reverse)
        c = 3.1 ** a
        assert c.val == 3.1
        assert c.gradients == ((a, 3.1 * np.log(3.1)),)

        # Node number and string (reverse), should raise an error
        with pytest.raises(TypeError):
            c = '3' ** a

    def test_elementary(self):
        # Test elementary functions for the Node type.
        # arcsin and arccos have specific domains and should raise
        # an error if the Node number is outside of the domain
        # range.
        a = Node(1)
        b = Node(0)

        # sin
        c = AD.sin(a)
        assert c.val == np.sin(1)
        assert c.gradients == ((a, np.cos(1)),)

        # cos
        c = AD.cos(a)
        assert c.val == np.cos(1)
        assert c.gradients == ((a, -np.sin(1)), )

        # tan
        c = AD.tan(a)
        assert c.val == np.tan(1)
        assert c.gradients == ((a, 1 / (np.cos(1) ** 2)), )

        # arcsin
        c = AD.arcsin(b)
        assert c.val == 0
        assert c.gradients == ((b, 1),)


        # arcsin with Node val not within -1 and 1, should raise an error
        with pytest.raises(ValueError):
            c = AD.arcsin(a)

        # arccos
        c = AD.arccos(b)
        assert c.val == np.arccos(0)
        assert c.gradients == ((b, -1), )

        # arccos with Node val not within -1 and 1, should raise an error
        with pytest.raises(ValueError):
            c = AD.arccos(a)

        # arctan
        c = AD.arctan(a)
        assert c.val == np.arctan(1)
        assert c.gradients == ((a, 1/2), )

        # exp
        c = AD.exp(a)
        assert c.val == np.exp(1)
        assert c.gradients == ((a, np.exp(1)), )

        # log
        c = AD.log(a, 2)
        assert c.val == 0
        assert c.gradients == ((a, 1/np.log(2)), )

        # log with string as base, should raise an error
        with pytest.raises(TypeError):
            c = AD.log(a, "base")

        # log with base lesser than or equal to zero, should raise an error
        with pytest.raises(ValueError):
            c = AD.log(a, 0)

        # log with Node val lesser than or equal to zero, should raise an error
        with pytest.raises(ValueError):
            c = AD.log(b, 2)

        # sqrt
        c = AD.sqrt(a)
        assert c.val == 1
        assert c.gradients == ((a, 1/2), )
        
        # square root of a Node number with the val part lesser than zero, should raise an error
        d = Node(-1)
        with pytest.raises(ValueError):
            c = AD.sqrt(d)

        # standard logistic
        c = AD.standard_logistic(a)
        assert c.val == 1 / (1 + np.exp(-1))
        assert c.gradients == ((a, 1/(1+np.exp(-1)) * (1-1/(1+np.exp(-1)))), )
        
        # sinh
        c = AD.sinh(a)
        assert c.val == np.sinh(1)
        assert c.gradients == ((a, np.cosh(1)),)

        # cosh
        c = AD.cosh(a)
        assert c.val == np.cosh(1)
        assert c.gradients == ((a, np.sinh(1)),)

        # tanh
        c = AD.tanh(a)
        assert c.val == np.tanh(1)
        assert c.gradients == ((a, 1/np.cosh(1)**2),)
