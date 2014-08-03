upstream

numpy.ndarray

class numpy.ndarray[source]

>An array object represents a multidimensional, homogeneous array of fixed-size items. An associated data-type object describes the format of each element in the array (its byte-order, how many bytes it occupies in memory, whether it is an integer, a floating point number, or something else, etc.)

>Arrays should be constructed using array, zeros or empty (refer to the See Also section below). The parameters given here refer to a low-level method (ndarray(...)) for instantiating an array.

>For more information, refer to the numpy module and examine the the methods and attributes of an array.

Parameters :

* (for the __new__ method; see Notes below)
* shape : tuple of ints
  * Shape of created array.
* dtype : data-type, optional
  * Any object that can be interpreted as a numpy data type.
* buffer : object exposing buffer interface, optional
  * Used to fill the array with data.
* offset : int, optional
  * Offset of array data in buffer.
* strides : tuple of ints, optional
  * Strides of data in memory.
* order : {‘C’, ‘F’}, optional
  * Row-major or column-major order.
See

Attributes

* T	Same as self.transpose(), except that self is returned if self.ndim < 2.
* data	Python buffer object pointing to the start of the array’s data.
* dtype	Data-type of the array’s elements.
* flags	Information about the memory layout of the array.
* flat	A 1-D iterator over the array.
* imag	The imaginary part of the array.
* real	The real part of the array.
* size	Number of elements in the array.
* itemsize	Length of one array element in bytes.
* nbytes	Total bytes consumed by the elements of the array.
* ndim	Number of array dimensions.
* shape	Tuple of array dimensions.
* strides	Tuple of bytes to step in each dimension when traversing an array.
* ctypes	An object to simplify the interaction of the array with the ctypes module.
* base	Base object if memory is from some other object.
