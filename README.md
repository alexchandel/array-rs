array-rs
========

A library that provides a multidimensional array object similar to NumPy's ndarray, along with routines for basic manipulation. array-rs is brand new, currently in the version 0.0.0.

## Background

As of Rust 0.11, there aren't any types in rust that provide the numerical computing functionality of ndarrays in Numpy. This package is designed to provide an `NDArray<T>` class to meet this need, and an Array typealias of `NDArray<f64>` for simplicity.

Presently, `NDArray` is typealiased to `Array`, and thus is only defined for the `f64` case, but eventually this will be extended to other data types. Although `f128` was recently removed from the language, due to its importance to scientific computing, ndarray-rs will explore ways to re-add this type.
