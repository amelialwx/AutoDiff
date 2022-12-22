import inspect
import numpy as np

from autodiff.ad import AD
from autodiff.node import Node

class ReverseMode(AD):
    """Reverse mode implementation based on nodes."""

    def get_gradients(node):
        """ 
        Compute the derivatives of `node` with respect to child nodes.

        Returns
        -------
        f'(x)
            The method returns the derivative(s) of `node` with respect to child nodes.
            
        """
        gradients = {}
        
        def compute_gradients(node, v):
            for child, gradient in node.gradients:
                v_child = v * gradient
                gradients[child] = gradients.get(child, 0) + v_child
                compute_gradients(child, v_child)
        
        compute_gradients(node, 1)
        return gradients
    
    def get_results(self, x):
        """
        Compute the value(s) and the derivative(s) of the function(s) based on input x.

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
            This method raises a `TypeError` if the type of input x is not supported.
            
        ValueError
            This method also raises a `ValueError` if the dimension of input x is not matched with the function(s).
            
        """
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
            jacobian = []
            vals = []
            for f in self.f:
                # initialize arguments list
                args = []
                nodes = []

                # get the function arguments
                function_args = inspect.getfullargspec(f)[0]
                  
                # if the function argument is an input, add it to the arguments list
                for i, input in enumerate(self.inputs):
                    if input in function_args:
                        node = Node(x[i])
                        args.append(node)
                        nodes.append(node)
                    # pad with 0 when the variable is not used in the function
                    else:
                        nodes.append(0)

                # unpack args and pass into f
                z = f(*args)
                gradients = ReverseMode.get_gradients(z)
                vals.append(z.val)
                j = []
                
                # fill jacobian with results
                for i, input in enumerate(self.inputs):
                    if input in function_args:
                        j.append(gradients[nodes[i]])
                    else:
                        j.append(0)
                jacobian.append(np.array(j))
                
            return np.array([np.array(vals), jacobian], dtype = object)
                    
        # if there is one function
        else:   
            # convert every element in args to a node
            args = [Node(arg) for arg in x]

            # unpack args and pass into f
            z = self.f(*args)
            gradients = ReverseMode.get_gradients(z)
            return np.array([z.val, np.array([gradients[node] for node in args])], dtype = object)