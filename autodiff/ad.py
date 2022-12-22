# File       : ad.py
# Description: Parent class AD that stores the function passed in by the user
#              to perform automatic differentiation on
import inspect
import numpy as np

class AD:
    """Automatic differentiation base class."""

    _supported_types = (int, float, np.ndarray, list)
    _supported_scalars = (int, float)
    _supported_vectors = (np.ndarray, list)

    def __init__(self, f, inputs=[]):
        """
        Initialize the function of which the derivative will be calculated based on input 'f'.

        Parameters
        ----------
        f : array-like
            Input with one or multiple functions.

        inputs : array-like
            List of input variables.
        """
        self.f = f
        self.inputs = inputs
        self.jacobian = False
        
        # check if user passed in a list-type of functions, if True, set jacobian to true
        if isinstance(self.f, self._supported_vectors):
            # check every function in list type is a function
            for f in self.f:
                if not inspect.isfunction(f):
                    raise TypeError(f"Unsupported type '{type(f)}'")
            # set self.jacobian to true
            self.jacobian = True
        # if the user passed in one function, check that it is a function
        elif not inspect.isfunction(self.f):
            raise TypeError(f"Unsupported type '{type(self.f)}'")
            
        # check if user passed in a list-type of variable(s)
        if not isinstance(self.inputs, self._supported_vectors):
            raise TypeError(f"Unsupported type '{type(self.inputs)}'")
        
        # convert inputs of supported type into a list
        self.inputs = list(self.inputs)
        
        # store the length of inputs
        self.n = len(self.inputs)

        # check that the input list is not empty
        if self.n == 0:
            raise ValueError("Input list is empty.")
        
        # check if every element in self.inputs is a string
        for i in self.inputs:
            if type(i) != str:
                raise TypeError(f"Unsupported type '{type(i)}' for input elements.")

        # check if every argument in the function(s) are present in the input
        if isinstance(self.f, self._supported_vectors):
            for f in list(self.f):
                function_args = inspect.getfullargspec(f)[0]
                for arg in function_args:
                    if arg not in self.inputs:
                        raise ValueError(f"Argument '{arg}' is not in '{self.inputs}'.")
        else:
            function_args = inspect.getfullargspec(self.f)[0]
            for arg in function_args:
                if arg not in self.inputs:
                    raise ValueError(f"Argument '{arg}' is not in '{self.inputs}'.")

    def get_function(self):
        """
        Get the function.

        Returns
        -------
        f
            The method returns the function 'f' .

        """
        return self.f

    def get_f(self, x):
        """
        Returns the value(s) of the function(s) evaluated at input 'x' computed by get_results.

        Parameters
        ----------
        x : Scalar, Vector. 
            The point at which the function(s) is evaluated. 
        
        Returns
        -------
        f(x)
            The method returns the value(s) of the function(s) evaluated at 'x'.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input 'x' is not supported.
            
        ValueError
            This method also raises a `ValueError` if the dimension of input 'x' is not matched with the function(s).

        """
        return self.get_results(x)[0]
    
    def get_f_prime(self, x):
        """
        Returns the derivative(s) of the function(s) based on input 'x' computed by get_results.

        Parameters
        ----------
        x : Scalar, Vector. 
            The point at which the derivative(s) of the function(s) is evaluated. 

        Returns
        -------
        f'(x)
            The method returns the derivative(s) of the function(s) at 'x'.

        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input 'x' is not supported.
            
        ValueError
            This method also raises a `ValueError` if the dimension of input 'x' is not matched with the function(s).
            
        """
        return self.get_results(x)[1]

    ### Square Root Function ###
    def sqrt(self):
        """
        Call the sqrt function in Dual or Node.
        """
        return self.__class__.sqrt(self)
    
    ### Exponential Function ###
    def exp(self):
        """
        Call the exp function in Dual or Node.
        """
        return self.__class__.exp(self)

    ### Logarithmic Function ###
    def log(self, base):
        """
        Call the log function in Dual or Node.
        """
        return self.__class__.log(self, base)
 
    ### Logistic Function ###
    def standard_logistic(self):
        """
        Call the standard_logistic function in Dual or Node.
        """
        return self.__class__.standard_logistic(self)

    ### Trigonometric Functions ### 
    def sin(self):
        """
        Call the sin function in Dual or Node.
        """
        return self.__class__.sin(self)
    
    def cos(self):
        """
        Call the cos function in Dual or Node.
        """
        return self.__class__.cos(self)
            
    def tan(self):
        """
        Call the cos function in Dual or Node.
        """
        return self.__class__.tan(self)

    ### Inverse Trigonometric Functions ###
    def arcsin(self):
        """
        Call the arcsin function in Dual or Node.
        """
        return self.__class__.arcsin(self)
    
    def arccos(self):
        """
        Call the arccos function in Dual or Node.
        """
        return self.__class__.arccos(self)
    
    def arctan(self):
        """
        Call the arctan function in Dual or Node.
        """
        return self.__class__.arctan(self)
    
    ### Hyperbolic Functions ###
    def sinh(self):
        """
        Call the sinh function in Dual or Node.
        """
        return self.__class__.sinh(self)
    
    def cosh(self):
        """
        Call the cosh function in Dual or Node.
        """
        return self.__class__.cosh(self)

    def tanh(self):
        """
        Call the tanh function in Dual or Node.
        """
        return self.__class__.tanh(self)
