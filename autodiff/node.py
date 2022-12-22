import numpy as np

class Node:
    """Node implementation for reversed mode."""

    _supported_scalars = (int, float)

    def __init__(self, val, gradients=()) -> None:
        """
        Initialize a node with its value and local gradients.

        Parameters
        ----------
        val : integer or float
            Value of a node.
            
        gradients : tuple
            Local gradients of a node.
            Consists of 1 or 2 tuples of (`child node`, `local gradient value`)

        """
        self.val = val
        self.gradients = gradients

    ### Elementary Functions ###
    def __add__(self, other):
        """
        Compute addition of two nodes or a node and a constant.

        Parameters
        ----------
        other : Node, constant
            Input object which is added to a node.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the addition.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input number other is not supported.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f"Unsupported type '{type(other)}'")

        if isinstance(other, self._supported_scalars):
            # scalar
            return Node(other + self.val, ((self, 1),))
        else:
            # node
            return Node(self.val + other.val, ((self, 1),(other, 1)))

    def __neg__(self):
        """
        Compute the negation of one node.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the negation.

        """
        return Node(-self.val, ((self, -1),))

    def __radd__(self, other):
        """
        Compute addition of two constants.

        Parameters
        ----------
        other : Constant
            Input constant which is added to another constant.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the addition.

        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Compute subtraction of one node from another node or a constant from a node.


        Parameters
        ----------
        other : Node, Constant
            Input object which is subtracted from a node.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the subtraction.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f"Unsupported type '{type(other)}'")

        if isinstance(other, self._supported_scalars):
            # scalar
            return Node(self.val - other, ((self, 1),))
        else:
            # node
            return Node(self.val - other.val, ((self, 1),(other, -1)))

    
    def __rsub__(self, other):
        """
        Compute subtraction of one constant from another constant.

        Parameters
        ----------
        other : Node, Constant
            Input object which is subtracted by a node.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the subtraction.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f"Unsupported type '{type(other)}'")

        return Node(other - self.val, ((self, -1),))

    def __mul__(self, other):
        """
        Compute multiplication of two nodes or a node and a constant.

        Parameters
        ----------
        other : Node, Constant
            Input object which is multiplied by a node.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the multiplication.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input number other is not supported.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            # scalar
            return Node(other * self.val, ((self, other),))
        else:
            # node
            return Node(self.val * other.val, ((self, other.val),(other, self.val)))
        
    def __rmul__(self, other):
        """
        Compute multiplication of two constants.

        Parameters
        ----------
        other : Constant
            Input constant which is multiplied by another constant.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the multiplication.

        """
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        Compute division of 'self' node by 'other' node.
        
        Parameters
        ----------
        other : Node, Constant
            Input object which divides a node.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the division.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            # scalar
            return Node(self.val/other, ((self, 1/other),))
        else:
            # node
            return Node(self.val/other.val, ((self, 1/other.val),(other, -self.val*other.val**-2)))
    
    def __rtruediv__(self, other):
        """
        Compute division of 'other' node by 'self' node.

        Parameters
        ----------
        other : Node, Constant
            Input object which is divided by a node.
        
        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the division.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        return Node(other/self.val, ((self, -other*self.val**-2),))    
    
    def __pow__(self, other):
        """
        Compute the exponentiation of raising one node value to the power of another node value or of one constant.

        Parameters
        ----------
        other : Node, Constant
            Input object to whose power the node will be raised.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the exponentiation.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input number other is not supported.

        """
        # check if other is of supported type
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            # scalar
            return Node(self.val**other, ((self, other*self.val**(other-1)),))
        else:
            # node
            return Node(self.val**other.val, ((self, other.val*self.val**(other.val-1)), 
                (other, self.val**other.val*np.log(self.val))))

    def __rpow__(self, other):
        """
        Compute the exponentiation of raising 'other' node value to the power of 'self' node value.

        Parameters
        ----------
        other : Node, Constant
            Input object which will be raised to the power of the 'self' node value.

        Returns
        -------
        node
            The method returns a new node initialized with its value and gradients resulting from the exponentiation.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input number other is not supported.

        """
        # check if other is of supported type
        if not isinstance(other, (*self._supported_scalars, Node)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        return Node(other**self.val, ((self, other**self.val*np.log(other)),))   

    ### Square Root Function ###
    def sqrt(self):
        """
        Compute the square root of one node.

        Returns
        -------
        Dual
            The method returns a new node initialized with its value and gradients resulting from the square root.
        
        Raises
        ------
        ValueError
            This method raises a `ValueError` if the value of input node value is less than zero.

        """
        if self.val < 0:
            raise ValueError("Cannot square root: value of node is less than 0.")
        return Node(self.val ** (1/2), ((self, 1/2*(self.val**(-1/2))),))
    
    ### Exponential Function ###
    def exp(self):
        """
        Compute the exponentiation of raising the natural number to the power of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the exponentiation.

        """
        return Node(np.exp(self.val), ((self, np.exp(self.val)),))

    ### Logarithmic Function ###
    def log(self, base):
        """
        Compute the logarithm to find the power to which the input base must be raised to yield the given node value.

        Parameters
        ----------
        base : Node, Constant
            Input base which is raised to yield a given node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the logarithm.
        
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input base number is not supported.
        
        ValueError
            This method raises a `ValueError` if the value of node or the value of input base is less than zero.

        """
        # check if base is of supported type
        if not isinstance(base, self._supported_scalars):
            raise TypeError(f"Unsupported base type '{type(base)}'")
        # check that the base is above 0.
        if base <= 0:
            raise ValueError("Cannot log: Base is less than or equal to 0.")
        # check that the value of the node is greater than 0.
        if self.val <= 0:
            raise ValueError("Cannot log: Value of node is less than or equal to 0.")
        return Node(np.log(self.val)/np.log(base), ((self, 1 / (np.log(base)*self.val)),))
 
    ### Logistic Function ###
    def standard_logistic(self):
        """
        Compute the value of the standard logistic function with the given node as input.

        Returns
        -------
        node
            The method returns the value of the standard logistic function with the given node as input.

        """
        return Node(1 / (1 + np.exp(-self.val)),((self, 1/(1+np.exp(-self.val)) * (1-1/(1+np.exp(-self.val)))),))

    ### Trigonometric Functions ### 
    def sin(self):
        """
        Compute the sine of a node value.

        Returns
        -------
        Node
            The method returns The method returns a new node initialized with its value and gradients resulting from the sine.

        """
        return Node(np.sin(self.val), ((self, np.cos(self.val)),))
    
    def cos(self):
        """
        Compute the cosine of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the cosine.
            
        """
        return Node(np.cos(self.val), ((self, -np.sin(self.val)),))
            
    def tan(self):
        """
        Compute the tangent of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the tangent.
            
        """
        return Node(np.tan(self.val), ((self, 1 / (np.cos(self.val) ** 2)),))

    ### Inverse Trigonometric Functions ###
    def arcsin(self):
        """
        Compute the arcsine of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the arcsine.
            
        Raises
        ------
        ValueError
            This method raises a `ValueError` if the value of the node is smaller than -1 or greater than 1.

        """
        if self.val >= 1 or self.val <= -1:
            raise ValueError("Value of node is not between -1 and 1.")
        return Node(np.arcsin(self.val), ((self, 1 / np.sqrt(1 - self.val ** 2)),))
    
    def arccos(self):
        """
        Compute the arccosine of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the arccosine.
            
        Raises
        ------
        ValueError
            This method raises a `ValueError` if the value of the node is smaller than -1 or greater than 1.

        """
        if self.val >= 1 or self.val <= -1:
            raise ValueError("Value of node is not between -1 and 1.")
        return Node(np.arccos(self.val), ((self, - 1 / np.sqrt(1 - self.val ** 2)),))
    
    def arctan(self):
        """
        Compute the arctangent of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the arctangent.

        """
        return Node(np.arctan(self.val), ((self, 1 / ((self.val ** 2) + 1)),))
    
    ### Hyperbolic Functions ###
    def sinh(self):
        """
        Compute the hyperbolic sine of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the hyperbolic sine.

        """
        return Node(np.sinh(self.val), ((self, np.cosh(self.val)),))
    
    def cosh(self):
        """
        Compute the hyperbolic cosine of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the hyperbolic cosine.

        """
        return Node(np.cosh(self.val), ((self, np.sinh(self.val)),))

    def tanh(self):
        """
        Compute the hyperbolic tangent of a node value.

        Returns
        -------
        Node
            The method returns a new node initialized with its value and gradients resulting from the hyperbolic tangent.

        """
        return Node(np.tanh(self.val), ((self, 1/np.cosh(self.val)**2),))
