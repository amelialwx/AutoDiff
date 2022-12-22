# File       : forwardmode.py
# Description: Forward mode implentation of automatic differentiation that 
#              uses the properties of dual numbers to return the value of
#              f(x) and f'(x)

import inspect
import numpy as np

from autodiff.ad import AD
from autodiff.dual import Dual

class ForwardMode(AD):
    """Forward mode implementation based on dual number data structure."""
    
    def get_results(self, x):
        """
        Compute the value(s) and the derivative(s) of the function(s) based on input 'x'.

        Parameters
        ----------
        x : Scalar, Vector. 
            The point at which the value(s) and derivative(s) of the function(s) are evaluated. 

        Returns
        -------
        f(x) and f'(x)
            The method returns both the value(s) and the derivative(s) of the function(s) at 'x'.
            
        Raises
        ------
        TypeError
            This method raises a `TypeError` if the type of input 'x' is not supported.
            
        ValueError
            This method also raises a `ValueError` if the dimension of input 'x' is not matched with the function(s).
            
        """
        reals = []
        duals = []
        
        # check that x is of supported type
        if not isinstance(x, self._supported_vectors):
            raise TypeError(f"Unsupported type '{type(x)}'")
            
        # check that x is 1-dimensional
        if len(np.shape(x)) != 1:
            raise ValueError(f"Input variables should be a 1-dimensional.")
            
        # convert x to a list
        x = list(x)
        
        # if there are multiple functions
        if self.jacobian:
            for f in self.f:
                # initialize arguments list
                args = []
                fduals = []
                
                # get the number of arguments
                n = len(inspect.getfullargspec(f)[0])
                        
                # if the number of arguments is lesser than the number of inputs
                if n < self.n:
                    # get the function arguments
                    function_args = inspect.getfullargspec(f)[0]
                    
                    # if the function argument is an input, add it to the arguments list
                    for i in self.inputs:
                        if i in function_args:
                            args.append(x[self.inputs.index(i)])
                    
                    # convert every element in args to a dual number with dual component 0 except for the target
                    for i in range(len(args)):
                        target_i = [Dual(arg, 0) for arg in args]
                        target_i[i] = Dual(args[i])
                        z = f(*target_i)
                        fduals.append(z.dual)
                    
                    # insert zeros for variables that are not present in the function
                    for i in self.inputs:
                        if i not in function_args:
                            fduals.insert(self.inputs.index(i), 0)
                            
                    duals.append(np.array(fduals))
                            
                # if the number of arguments is equal to the number of inputs
                else:
                    # set the arguments to be the inputs
                    args = x
                    
                    # convert every element in args to a dual number with dual component 0 except for the target
                    for i in range(len(args)):
                        target_i = [Dual(arg, 0) for arg in args]
                        target_i[i] = Dual(args[i])
                        z = f(*target_i)
                        fduals.append(z.dual)

                    duals.append(np.array(fduals))

                # convert every element in args to a dual number with dual component 1
                args = [Dual(arg) for arg in args]
                
                # unpack args and pass into f
                z = f(*args)
                reals.append(z.real)
                
            return np.array([np.array(reals), duals], dtype = object)
                    
        # if there is one function
        else:   
            # get the number of arguments
            n = len(inspect.getfullargspec(self.f)[0])
            
            # convert every element in args to a dual numebr with dual component 0 except for the target
            for i in range(len(x)):
                target_i = [Dual(arg, 0) for arg in x]
                target_i[i] = Dual(x[i])
                z = self.f(*target_i)
                duals.append(z.dual) 
            
            # convert every element in args to a dual number with dual component 1
            args = [Dual(arg) for arg in x]
            
            # unpack args and pass into f
            z = self.f(*args)
            reals = z.real  
            
            return np.array([reals, np.array(duals)], dtype = object)
