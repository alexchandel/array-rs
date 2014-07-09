array-rs ![Build Status](https://travis-ci.org/alexchandel/array-rs.png)
========

A library that provides a multidimensional array object similar to NumPy's ndarray, along with routines for basic manipulation. array-rs is brand new, currently in the version 0.0.1.

## Background

As of Rust 0.11, there aren't any types in rust that provide the numerical computing functionality of ndarrays in Numpy. This package is intends to provide an `NDArray<T>` class to meet this need. Array is equivalent to `NDArray<f64>`.

Only `Array` is defined at the moment, but eventually the `NDArray` generic will be written allowing other data types. Although `f128` was recently removed from the language, due to its importance to scientific computing, an `NDArray<f128>` type is still desirable.


## Use

Please test the Array class and report any issues to [array-rs/issues](https://github.com/alexchandel/array-rs/issues)!

## Development

The API should be kept as simple as possible for now, and provide only basic array manipulation. The goal is to have a stable, tested `Array` class. After this, some numerical functionality could be added in a separate module.
