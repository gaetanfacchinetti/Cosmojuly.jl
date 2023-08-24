# Cosmojuly

New cosmology package written in Julia. Complementary to other tools developped in parallel, the idea here is to provide a simple autonomous package that can compute dark matter related quantities from analytical models. The main goal will be to implement the FSL model developped in [[arXiv:1610.02233](https://arxiv.org/abs/1610.02233), [arXiv:2201.09788](https://arxiv.org/abs/2201.09788), [thesis](https://theses.hal.science/tel-03414834/)] and add new features.

Would you find this package during his developement and decide to use it, please credit it. If the code is already mature enough and contains (at least bits of) implemented routines for the FSL model, then cite the three references mentionned above.

### Documentation

A documentation (under construction) is available [here](https://gaetanfacchinetti.github.io/docs/Cosmojuly.jl/index.html).

### Why Julia?

A first version of this code has been developped in C/C++. However, the language and the implementation of the model did not allow us to make this package user friendly. For the second version, Julia appeared to be the most sensible choice. A python package would have been readily written in comprehensive way but it would also have been too slow to properly run on a laptop. Julia combines both C/C++ strengths for optimization and Python strengths for legibility.

### A glimpse on the to do list:

- [x] write the basis of a general cosmology module
- [x] implement the power spectrum module
- [x] implement a mass function module based on excursion sets
- [ ] implement a dark halo module
- [ ] implement the FSL module
- [ ] implement the Eddington inversion routines
- [ ] ...


### Contributing to the project

Feel free to contribute to the project. Any help is welcome. Please follow the Julia recommendation for documenting the code.
