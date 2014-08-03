array-rs ![Build Status](https://travis-ci.org/alexchandel/array-rs.png)
========

A library that provides a multidimensional array object similar to NumPy's ndarray, along with routines for basic manipulation. array-rs is brand new, currently in the version 0.0.1.

## Background

Very much a *WIP*! Stay tuned / send suggestions!

As of Rust 0.12, there aren't any types in rust that provide the numerical computing functionality of ndarrays in Numpy. array-rs intends to provide a generic `NDArray<T>` class. It also provides the type Array,equivalent to `NDArray<f64>`.

Only `Array` is defined at the moment, but eventually the `NDArray` generic will be written, allowing other data types.

## Use

Please test the Array class and report any issues to [array-rs/issues](https://github.com/alexchandel/array-rs/issues)!

## Development

* The API should be kept as simple as possible for now, and provide only basic array manipulation. The goal is to have a stable, tested `Array` class.

* Numerical functionality can be added in a separate module.

* Although `f128` was recently removed from the language, due to its importance to scientific computing, an `NDArray<f128>` type is still desirable.

* SIMD instructions for speedup
