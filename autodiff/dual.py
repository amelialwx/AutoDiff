# File       : dual.py
# Description: Dual class to store dual numbers and overload operators and 
#              functions to perform basic artihmetic
import numpy as np

class Dual():
    """Dual number implementation to perform basic arithmetic and geometric operations."""
    
    _supported_scalars = (int, float)
    
    def __init__(self, real, dual = 1.0):
        """
        Initialize a dual number based on inputs 'real' and 'dual'.

        Parameters
        ----------
        real : integer or float
            Input number to initialize the real part of a dual number.
            
        dual : integer or float
            Input number to initialize the dual part of a dual number.

        """
        self.real = real
        self.dual = dual
    
    ### Elementary Functions ###
    def __add__(self, other):
        """
        Compute the addition of two dual numbers or of one dual number and one real number.

        Parameters
        ----------
        other : Dual, Scalar
            Input number which is added to a dual number.

        Returns
        -------
        Dual
            The method returns the value of adding two dual numbers or one dual number and one real number.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input number 'other' is not supported.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Dual)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            # scalar
            return Dual(other + self.real, self.dual)
        else:
            # dual
            return Dual(self.real + other.real, self.dual + other.dual)
        
    def __radd__(self, other):
        """
        Overload the addition function to compute the addition of two real numbers.

        Parameters
        ----------
        other : Scalar
            Input number which is added to a real number.

        Returns
        -------
        Dual
            The method returns the value of adding two real numbers.
        
        """
        return self.__add__(other)
    
    def __sub__(self, other):
        """
        Compute the subtraction of 'other' dual number from 'self' dual number.

        Parameters
        ----------
        other : Dual, Scalar
            Input number which is subtracted from a dual number.

        Returns
        -------
        Dual
            The method returns the resulting value of subtaction.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input number other is not supported.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Dual)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            # scalar
            return Dual(self.real - other, self.dual)
        else:
            # dual
            return Dual(self.real - other.real, self.dual - other.dual)
        
    def __rsub__(self, other):
        """
        Compute the subtraction of 'self' dual number from 'other' dual number.

        Parameters
        ----------
        other : Dual, Scalar
            Input number which is subtracted by a dual number.

        Returns
        -------
        Dual
            The method returns the resulting value of subtaction.

        """
        return -self + other
    
    def __mul__(self, other):
        """
        Compute the multiplication of two dual numbers or of one dual number and one real number.

        Parameters
        ----------
        other : Dual, Scalar
            Input number which is multiplied by a dual number.

        Returns
        -------
        Dual
            The method returns the value of multiplying two dual numbers or of one dual number and one real number.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input number other is not supported.

        """
        # check if other is of a supported type
        if not isinstance(other, (*self._supported_scalars, Dual)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            # scalar
            return Dual(other * self.real, other * self.dual)
        else:
            # dual
            real = self.real * other.real
            dual = self.real * other.dual + self.dual * other.real
            return Dual(real, dual)
        
    def __rmul__(self, other):
        """
        Overload the multiplication function to compute the multiplication of two real numbers.

        Parameters
        ----------
        other : Scalar
            Input number which is multiplied by a real number.

        Returns
        -------
        Dual
            The method returns the value of multiplying two real numbers.
        
        """
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        Compute the division of 'self' dual number by 'other' dual number.

        Parameters
        ----------
        other : Dual, Scalar
            Input number which divides a dual number.

        Returns
        -------
        Dual
            The method returns the value of the division.

        """
        return self * other ** (-1)
    
    def __rtruediv__(self, other):
        """
        Overload the division function to compute the divition of one real number by another real number.

        Parameters
        ----------
        other : Scalar
            Input number which divides a real number.

        Returns
        -------
        Dual
            The method returns the value of dividing one real number by another real number.
        
        """
        return other * self ** (-1)
        
    def __neg__(self):
        """
        Compute the negation of one dual number.

        Returns
        -------
        Dual
            The method returns the value of the negation of one dual number.

        """
        return Dual(-self.real, -self.dual)
        
    def __pow__(self, other):
        """
        Compute the exponentiation of raising one dual number to the power of another dual number or of one real number.

        Parameters
        ----------
        other : Dual, Scalar
            Input exponent to which the base dual number will be raised.

        Returns
        -------
        Dual
            The method returns the value of raising one dual number to the power of another dual number or of one real number.
        
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input exponent other is not supported.

        """
        # check if other is of supported type
        if not isinstance(other, (*self._supported_scalars, Dual)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            # scalar
            real = self.real ** other
            dual = self.dual * other * (self.real ** (other - 1))
            return Dual(real, dual)
        else:
            # dual
            real = self.real ** other.real
            dual = other.real * self.dual * self.real ** (other.real - 1) + np.log(self.real) * other.dual * self.real ** other.real
            return Dual(real, dual)
        
    def __rpow__(self, other):
        """
        Overload the exponentiation function to compute the exponentiation of raising a real number to the power of another real 
        number.

        Parameters
        ----------
        other : Scalar
            Input exponent to which the base real number will be raised.

        Returns
        -------
        Dual
            The method returns the value of raising one real number to the power of another real number.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input exponent other is not supported.

        """
        # check if other is of supported type
        if not isinstance(other, self._supported_scalars):
            raise TypeError(f"Unsupported type '{type(other)}'")
        real = other ** self.real
        dual = (other ** self.real) * np.log(other) * self.dual
        return Dual(real, dual)
    
    ### Square Root Function ###
    def sqrt(self):
        """
        Compute the square root of one dual number.

        Returns
        -------
        Dual
            The method returns the square root of one dual number.
        
        Raises
        ------
        ValueError
            This method raises a `ValueError` if the value of input dual number is less than zero.

        """
        # check that the real component of the dual number is greater than or equal to 0.
        if self.real < 0:
            raise ValueError("Cannot square root: real part of dual number is lesser than 0.")
        return Dual(self.real ** (1/2), self.dual * ((1/2) * (self.real ** (-1/2))))
    
    ### Exponential Function ### 
    def exp(self):
        """
        Compute the exponentiation of raising the natural number to the power of one dual number.

        Returns
        -------
        Dual
            The method returns the value of raising the natural number to the power of one dual number.

        """
        return Dual(np.exp(self.real), np.exp(self.real) * self.dual)
    
    ### Logarithmic Function ###
    def log(self, base):
        """
        Compute the logarithm to find the power to which the input base must be raised to yield the given dual number.

        Parameters
        ----------
        base : Dual, Scalar
            Input base which is raised to yield a given dual number.

        Returns
        -------
        Dual
            The method returns the value of the exponent to which the input base must be raised to yield the given dual number.
        
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input base number is not supported.
        
        ValueError
            This method raises a `ValueError` if the real part of the dual number or the value of input base is less than zero.

        """
        # check if base is of supported type
        if not isinstance(base, self._supported_scalars):
            raise TypeError(f"Unsupported base type '{type(base)}'")
        # check that the base is above 0.
        if base <= 0:
            raise ValueError("Cannot log: Base is lesser than or equal to 0.")
        # check that the real component of the dual number is above 0.
        if self.real <= 0:
            raise ValueError("Cannot log: Real part of the dual number is lesser than or equal to 0.")
        return Dual(np.log(self.real) / np.log(base), 1 / (np.log(base) * self.real) * self.dual)
    
    ### Logistic Function ###
    def standard_logistic(self):
        """
        Compute the value of the standard logistic function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the standard logistic function with the given dual number as input parameter.

        """
        return 1 / (1 + Dual.exp(-self))
    
    ### Trigonometric Functions ### 
    def sin(self):
        """
        Compute the value of the sine function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the sine function with the given dual number as input parameter.

        """
        return Dual(np.sin(self.real), np.cos(self.real) * self.dual)
    
    def cos(self):
        """
        Compute the value of the cosine function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the cosine function with the given dual number as input parameter.

        """
        return Dual(np.cos(self.real), np.sin(self.real) * -self.dual)
    
    def tan(self):
        """
        Compute the value of the tangent function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the tangent function with the given dual number as input parameter.

        """
        return Dual(np.tan(self.real), self.dual / (np.cos(self.real) ** 2))
    
    ### Inverse Trigonometric Functions ###
    def arcsin(self):
        """
        Compute the value of the arcsine function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the arcsine function with the given dual number as input parameter.
            
        Raises
        ------
        ValueError
            This method raises a `ValueError` if the real part of the dual number is smaller than -1 or greater than 1.

        """
        # check that the real component of the dual number is between -1 and 1.
        if self.real >= 1 or self.real <= -1:
            raise ValueError("Cannot arcsin: Real part of dual number is not between -1 and 1.")
        return Dual(np.arcsin(self.real), self.dual / np.sqrt(1 - self.real ** 2))
    
    def arccos(self):
        """
        Compute the value of the arccosine function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the arccosine function with the given dual number as input parameter.
            
        Raises
        ------
        ValueError
            This method raises a `ValueError` if the real part of the dual number is smaller than -1 or greater than 1.

        """
        # check that the real component of the dual number is between -1 and 1.
        if self.real >= 1 or self.real <= -1:
            raise ValueError("Cannot arccos: Real part of dual number is not between -1 and 1.")
        return Dual(np.arccos(self.real), - self.dual / np.sqrt(1 - self.real ** 2))
    
    def arctan(self):
        """
        Compute the value of the arctangent function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the arctangent function with the given dual number as input parameter.
            
        """
        return Dual(np.arctan(self.real), self.dual / ((self.real ** 2) + 1))
    
    ### Hyperbolic Functions ###
    def sinh(self):
        """
        Compute the value of the hyperbolic sine function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the hyperbolic sine function with the given dual number as input parameter.
            
        """
        return (self.exp() - (Dual.exp(-self))) / 2
    
    def cosh(self):
        """
        Compute the value of the hyperbolic cosine function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the hyperbolic cosine function with the given dual number as input parameter.
            
        """
        return (self.exp() + (Dual.exp(-self))) / 2
        
    def tanh(self):
        """
        Compute the value of the hyperbolic tangent function with the given dual number as input parameter.

        Returns
        -------
        Dual
            The method returns the value of the hyperbolic tangent function with the given dual number as input parameter.
            
        """
        return self.sinh() / self.cosh()
