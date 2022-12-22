# An Automatic Differentiation Library

![test.yml](https://github.com/amelialwx/AutoDiff/actions/workflows/test.yml/badge.svg)
![coverage.yml](https://github.com/amelialwx/AutoDiff/actions/workflows/coverage.yml/badge.svg)

An automatic differentiation library for my CS207 final project at Harvard University.


## What is autodiff?

autodiff is a package that provides users with forward-mode and reverse-mode automatic differentiation tools to compute derivatives of mathematical functions up to machine precision with high efficiency. With this software, the user can perform automatic differentiation for themselves. Our software focuses on the elementary functions of automatic differentiation for users who are more unfamiliar with the concept to utilize.

The library currently supports forward mode and reverse mode automatic differentiation, real functions with scalar or vector inputs, and multiple functions with scalar or vector inputs. 

Forward mode is implemented using the unique properties of dual numbers. Reverse mode is implemented using reverse traversal through a computational graph.

## How to Install

The user can install the package using the code below

```{python}
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple AutoDiff-Library==1.0
```

## How to Use

Please take a look at the [documentation](https://github.com/amelialwx/AutoDiff/blob/main/docs/documentation.ipynb) under the "How to Use ```autodiff```" section.

## Extra

Code coverage results can be found [here](https://amelialwx.github.io/AutoDiff/).

Package on Test PyPI can be found [here](https://test.pypi.org/project/AutoDiff-Library/1.0/).



