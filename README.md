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

## Author
[![Contributors](https://contrib.rocks/image?repo=amelialwx/AutoDiff)](https://github.com/amelialwx/AutoDiff/graphs/contributors)

[![GitHub](https://img.shields.io/badge/GitHub-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/amelialwx)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=github&logoColor=white)](https://linkedin.com/in/amelialwx)
[![Website](https://img.shields.io/badge/website-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://amelialwx.github.io)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:weixili@g.harvard.edu)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://amelialwx.medium.com)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/dplyrr)

