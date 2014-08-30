use std::iter::Iterator;
use std::iter::Extendable; // for Array construction
use std::vec::Vec;
use std::num::{Zero, One};

// for serialization:
use std::path::Path;
use std::io::IoResult;
use std::io::fs::File;
use std::io::Open;
use std::io::Write;

// Array functionality, including the N-dimensional array.

/// A multidimensional array. The last index is the column index, and the
/// second-last is the row index.
/// TODO generic NDArray<T> for ints, floats including f128, and bool
#[deriving(Show)]
pub struct Array
{
	_inner: Vec<f64>,	// must never grow!
	_shape: Vec<uint>,	// product equals _inner.len()
	_size: uint			// equals product of _shape and _inner.len()
}

// TODO This typedef is backwards! Make Array = NDArray<T>!
type NDArray = Array;

/// A utility function for the NDArray implementation.
trait Dot {
	fn dot(left: Vec<Self>, right: Vec<Self>) -> Self;
}

impl Dot for f64 {
	#[inline]
	fn dot(left: Vec<f64>, right: Vec<f64>) -> f64
	{
		left.iter().zip(right.iter()).fold(0f64, |b, (&a1, &a2)| b + a1*a2)
	}
}

trait Prod {
	fn prod(vector: &Vec<Self>) -> Self;
}

impl Prod for f64
{
	#[inline]
	fn prod(vector: &Vec<f64>) -> f64
	{
		vector.iter().fold(1f64, |b, &a| a * b)
	}
}

impl Array
{
	/*			CONSTRUCTORS			*/

	/// Constructs a 3x3 demo array.
	pub fn square_demo() -> Array
	{
		let p = Array::with_slice(&[0f64,1f64,2f64,3f64,4f64,5f64,6f64,7f64,8f64]);
		let q = p.reshape(&[3,3]);
		q
	}

	/// Constructs a 0-dimensional array.
	pub fn empty() -> Array
	{
		Array {
			_inner: Vec::new(),
			_shape: vec!(0),
			_size: 0
		}
	}

	/// Constructs a singleton array with 1 element.
	pub fn with_scalar(scalar: f64) -> Array
	{
		Array {
			_inner: vec!(scalar),
			_shape: vec!(1),
			_size: 1
		}
	}

	/// Constructs a 1D array with contents of slice.
	pub fn with_slice(sl: &[f64]) -> Array
	{
		let capacity = sl.len();
		Array {
			_inner: sl.to_vec(), // TODO is this correct?
			_shape: vec!(capacity),
			_size: capacity
		}
	}

	// TODO pub fn with_slice_2(sl: &[[f64]]) -> Array
	// TODO pub fn with_slice_3(sl: &[[[f64]]]) -> Array
	// TODO etc

	/// Constructs a 1D array with contents of Vec.
	pub fn with_vec(vector: Vec<f64>) -> Array
	{
		let capacity = vector.len();
		Array {
			_inner: vector.clone(),
			_shape: vec!(capacity),
			_size: capacity
		}
	}

	/// Constructs an array with given vector's contents, and given shape.
	pub fn with_vec_shape(vector: Vec<f64>, shape: &[uint]) -> Array
	{
		let array = Array::with_vec(vector);
		array.reshape(shape);
		array
	}

	/// Internally allocates an ND array with given shape.
	fn alloc_with_shape(shape: &[uint]) -> Array
	{
		let capacity = shape.iter().fold(1u, |b, &a| a * b);
		Array {
			_inner: Vec::with_capacity(capacity),
			_shape: shape.to_vec(),
			_size: capacity
		}
	}

	/// Internally allocates an ND array with shape of given array
	fn alloc_with_shape_of(other: &Array) -> Array
	{
		assert!(other._inner.len() == other._size, "Array is inconsistent.");
		Array {
			_inner: Vec::with_capacity(other._size),
			_shape: other._shape.clone(),
			_size: other._size
		}
	}

	/// Constructs an array of zeros, with given shape.
	pub fn zeros(shape: &[uint]) -> Array
	{
		let mut a = Array::alloc_with_shape(shape);
		a._inner.grow(a._size, &0f64);
		a
	}

	/// Constructs an array of ones, with given shape.
	pub fn ones(shape: &[uint]) -> Array
	{
		let mut a = Array::alloc_with_shape(shape);
		a._inner.grow(a._size, &1f64);
		a
	}

	/*			SIZE			*/

	/// Returns a slice containing the array's dimensions.
	#[inline]
	pub fn shape<'a>(&'a self) -> &'a [uint]
	{
		self._shape.as_slice()
	}

	/// Returns the number of dimensions of the array.
	#[inline]
	pub fn ndim(&self) -> uint
	{
		self._shape.len()
	}

	/// Returns the number of elements in the array. This is equal to the
	/// product of the array's shape
	pub fn size(&self) -> uint
	{
		self._shape.iter().fold(1u, |b, &a| a * b)
	}

	/*			ELEMENT ACCESS			*/
	fn index_to_flat(&self, index: &[uint]) -> uint
	{
		let offset: uint = *index.last().unwrap();
		let ind_it = index.iter().take(index.len()-1);
		let shp_it = self._shape.iter().skip(1);
		ind_it.zip(shp_it).fold(offset,
			|b, (&a1, &a2)| b + a1*a2)
	}


	/// Gets the n-th element of the array, wrapping over rows/columns/etc.
	#[inline]
	pub fn get_flat(&self, index: uint) -> f64
	{
		*self._inner.get(index)
	}

	/// Gets a scalar element, given a multiindex. Fails if index isn't valid.
	pub fn get(&self, index: &[uint]) -> f64
	{
		assert!(index.len() == self._shape.len())
		self.get_flat(self.index_to_flat(index))
	}

	/// Gets a particular row-vector, column-vector, page-vector, etc.
	/// The index for the axis dimension is ignored.
	/// TODO DST
	pub fn get_vector_axis(&self, index: &[uint], axis: uint) -> Vec<f64>
	{
		let mut slice = index.to_vec();
		let mut it = range(0u, *self._shape.get(axis)).map(|a| {
			*slice.get_mut(axis) = a;
			self.get(slice.as_slice())
		});
		it.collect()
	}

	/// Gets a particular row-vector, column-vector, page-vector, etc.
	/// Specify (-1 as uint) for the desired dimension! Fails otherwise.
	pub fn get_vector(&self, index: &[uint]) -> Vec<f64>
	{
		let axis: uint = *index.iter().find(|&&a| (a == -1i as uint)).unwrap();
		self.get_vector_axis(index, axis)
	}

	// TODO sub-array iteration


	/*			ELEMENT ASSIGNMENT			*/

	pub fn set_flat(&mut self, index: uint, value: f64)
	{
		*self._inner.get_mut(index) = value;
	}

	pub fn set(&mut self, index: &[uint], value: f64)
	{
		assert!(index.len() == self._shape.len());
		let flat = self.index_to_flat(index);
		self.set_flat(flat, value);
	}

	// TODO single-element assignment
	// TODO sub-array assignment



	/*			BROADCASTING OPERATIONS			*/
	/// Returns array obtained by applying function to every element.
	pub fn apply(&self, func: fn(f64) -> f64) -> Array
	{
		let it = self._inner.iter().map(|&a| func(a));
		let mut empty_array = Array::alloc_with_shape_of(self);
		empty_array._inner.extend(it);
		empty_array
	}

	/*			SHAPE OPERATIONS			*/

	/// Returns a new array with shifted boundaries. Fails if sizes differ.
	/// ATM, this trivially reflows the the array.
	/// TODO offer option to pad with zeros?
	pub fn reshape(&self, shape: &[uint]) -> Array
	{
		/* If size == product of new lengths, assign
		 * else if size is divisible by product, add one dimension, and assign
		 * else fail.
		 */
		let capacity = shape.iter().fold(1u, |b, &a| a * b);
		assert!(self._size % capacity == 0,
			"Array::reshape: self.size must be divisible by shape.prod()!")
		if self._size == capacity
		{
			Array {
				_inner: self._inner.clone(),
				_shape: shape.to_vec(),
				_size: self._size
			}
		}
		else /* self._size % capacity == 0 */
		{
			let mut full_shape = shape.to_vec();
			full_shape.push(self._size / capacity);
			Array {
				_inner: self._inner.clone(),
				_shape: full_shape,
				_size: self._size
			}
		}
	}

	/*			CONTRACTING OPERATIONS			*/

	/// like AdditiveIterator
	/// Currently sums every element in array
	pub fn sum(&self /*, axis/axes */) -> Array
	{
		let ref v: Vec<f64> = self._inner;
		let sum_f = v.iter().fold(0f64, |b, &a| a + b);
		Array {
			_inner: vec!(sum_f),
			_shape: vec!(1),
			_size: 1
		}
	}

	/// like MultiplicativeIterator
	/// Currently multiplies every elment in array
	pub fn prod(&self /*, axis/axes */) -> Array
	{
		let prod_f = Prod::prod(&self._inner);
		Array {
			_inner: vec!(prod_f),
			_shape: vec!(1),
			_size: 1
		}
	}

	/// Dot product
	/// Array dotting essentially treats NDArrays as collections of matrices.
	/// Dot acts on last axis (row index) of LHS, and second-last axis of RHS
	/// --> dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
	///
	/// TODO if arrays aren't identically shaped (except last 2 dims)
	pub fn dot(&self, rhs: &Array) -> Array
	{
		let l_ndim: uint = self.ndim();
		let r_ndim: uint = rhs.ndim();
		assert!(l_ndim > 0 && r_ndim > 0, "Cannot dot an empty array!");

		let v_len: uint = *self._shape.get(l_ndim-1);

		if l_ndim == 1 && r_ndim == 1
		{
			if r_ndim == 1
			{
				assert!(v_len == *rhs._shape.get(r_ndim-1),
					"Lengths of dottend vectors must be the same!");
				let dot_f = self._inner.iter().zip(rhs._inner.iter()).fold(0f64,
					|b, (&a1, &a2)| b + a1*a2);
				Array {
					_inner: vec!(dot_f),
					_shape: vec!(1),
					_size: 1
				}
			}
			else /* LHS := row vector, RHS := column vectors or matrix */
			{
				assert!(v_len == *rhs._shape.get(r_ndim-2),
					"Row length of LHS must equal column length of RHS!");

				let mut new_dim: Vec<uint> = rhs.shape().slice(0, r_ndim-2).to_vec();
				new_dim.push(1); // Only 1 column
				new_dim.push(*self.shape().get(r_ndim-1).unwrap());
				let new_size = new_dim.iter().fold(1u, |b, &a| a * b);
				let mut new_inner = Vec::with_capacity(new_size);
				new_inner.push(1f64);
				// Do half of iteration below.

				fail!(":( This should be incorporated into array broadcasting");
			}
		}
		else /* LHS := row vectors or matrix */
		{
			assert!(r_ndim > 1, "RHS must have column vectors, consider transposing.");
			assert!(v_len == *rhs._shape.get(r_ndim-2),
					"Row length of LHS must equal column length of RHS!");
			assert!(self.shape().as_slice().slice(0, l_ndim-1) ==
					rhs.shape().as_slice().slice(0, r_ndim-1),
					"All dimensions of RHS and LHS except last 2 must be equal!");
			// ^^^^ at the moment, we only support identic
			// TODO broadcast smaller array onto larger one!

			// All but last 2 dims of both,
			// 2nd last dim of left,
			// last dim of right
			let mut new_dim: Vec<uint> = self.shape().slice(0, l_ndim-1).to_vec();
			new_dim.push(*self.shape().get(r_ndim-1).unwrap());
			let new_size = new_dim.iter().fold(1u, |b, &a| a * b);
			let mut new_inner = Vec::with_capacity(new_size);

			// let iters = new_dim.iter().map(|length| range(0u, length));
			// TODO idiomatic way to tensor-multiply the iterators
			let offsets: Vec<uint> = {
				let mut _offset = vec!(1u);
				for d in new_dim.iter().rev().take(new_dim.len()-1)
				{
					let _last_offset = *_offset.last().unwrap();
					_offset.push(d * _last_offset)
				}
				_offset.reverse(); // WARNING expensive
				_offset
			};

			// if new_dim = [2, 2, 3, 3], two stacks of two pages of 3x3 matrices
			// then offsets = [2*3*3, 3*3, 3, 1]
			// dot a multi_index with an offsets to obtain a flat_index
			// equivalently, flat_index % offset[i] / new_dim[i] == multi_index[i]

			// iterate over elements of new array
			for i in range(0u, new_size)
			{
				let multi_index: Vec<uint> = offsets.iter().zip(new_dim.iter()).map(
					|(&offset, &dim)| i / offset % dim).collect();
				let l_axis = l_ndim-1;
				let r_axis = r_ndim-2;
				let mut l_index = multi_index.clone();
				let mut r_index = multi_index.clone();
				// let new_val = vl.zip(vr).fold(0f64, |b, (&l, &r)| b + l*r);
				let mut new_val: f64 = 0f64;
				for j in range(0u, v_len)
				{
					*l_index.get_mut(l_axis) = j; // row vectors from left
					*r_index.get_mut(r_axis) = j; // col vectors from right
					new_val += self.get(l_index.as_slice()) * rhs.get(r_index.as_slice());
				}
				new_inner.push(new_val);
			}

			Array {
				_inner: new_inner,
				_shape: new_dim,
				_size: new_size
			}
		}
	}

	/// Writes the array into the specified file.
	/// TODO is this idiomatic?
	pub fn write_to(&self, path: &str) -> IoResult<()>
	{
		match File::open_mode(&Path::new(path), Open, Write)
		{
			Ok(mut file) => {
				match file.write_le_u64(self._shape.len() as u64)
				{
					Ok(_) => {
						for &dim in self._shape.iter()
						{
							match file.write_le_u64(dim as u64) {
								Ok(_) => (),
								Err(io_err) => return Err(io_err)
							}
						}
						match file.write_le_u64(self._inner.len() as u64)
						{
							Ok(_) => {
								for &element in self._inner.iter()
								{
									match file.write_le_f64(element) {
										Ok(_) => (),
										Err(io_err) => return Err(io_err)
									}
								}
								file.write_u8(0)
							},
							Err(io_err) => Err(io_err)
						}
					},
					Err(io_err) => Err(io_err)
				}
			},
			Err(io_err) => Err(io_err)
		}
	}

	/// Writes the array into the specified file.
	/// TODO is this idiomatic?
	pub fn read_from(path: &str) -> IoResult<Array>
	{
		match File::open(&Path::new(path))
		{
			Ok(mut file) => {
				match file.read_le_u64() {
					Ok(ndim) => {
						let mut _shape = Vec::with_capacity(ndim as uint);
						for _ in range(0u, ndim as uint) {
							match file.read_le_u64() {
								Ok(len) => _shape.push(len as uint),
								Err(io_err) => return Err(io_err)
							}
						}
						match file.read_le_u64() {
							Ok(_size) => {
								let mut _inner = Vec::with_capacity(_size as uint);
								for _ in range(0u, _size as uint) {
									match file.read_le_f64() {
										Ok(element) => _inner.push(element),
										Err(io_err) => return Err(io_err)
									}
								}
								Ok(Array {
									_inner: _inner,
									_shape: _shape,
									_size: _size as uint
								})
							},
							Err(io_err) => return Err(io_err)
						}
					},
					Err(io_err) => return Err(io_err)
				}
			},
			Err(io_err) => return Err(io_err)
		}
	}

	// TODO broadcasting of fn(f64) functions

	// TODO pretty printing of arrays
}

/* TODO subarray indexing */
// impl Index<uint, Array> for Array {
// 	fn index(&self, index: &uint) -> Array
// 	{
// 		self._inner.get(index)
// 	}
// }

/// Broadcast addition over all elements. Fail if shapes differ.
impl Add<Array, Array> for Array {
	fn add(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::alloc_with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			p._inner.extend(arg_it.map(|(&a1, &a2)|
				a1 + a2
			));
			p
		}
	}
}

impl Sub<Array, Array> for Array {
	fn sub(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::alloc_with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			p._inner.extend(arg_it.map(|(&a1, &a2)|
				a1 - a2
			));
			p
		}
	}
}

impl Mul<Array, Array> for Array {
	fn mul(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::alloc_with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			p._inner.extend(arg_it.map(|(&a1, &a2)|
				a1 * a2
			));
			p
		}
	}
}

impl Div<Array, Array> for Array {
	fn div(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::alloc_with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			p._inner.extend(arg_it.map(|(&a1, &a2)|
				a1 / a2
			));
			p
		}
	}
}

impl Rem<Array, Array> for Array {
	fn rem(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::alloc_with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			p._inner.extend(arg_it.map(|(&a1, &a2)|
				a1 % a2
			));
			p
		}
	}
}

impl Neg<Array> for Array {
	fn neg(&self) -> Array
	{
		let mut p = Array::alloc_with_shape(self.shape());
		let arg_it = self._inner.iter();
		p._inner.extend(arg_it.map(|&a1|
			-a1
		));
		p
	}
}

// impl Zero for Array {
// 	fn zero() -> Array {
// 		Array::zeros(&[0])
// 	}

// 	fn is_zero(&self) -> bool {

// 	}
// }

impl One for Array {
	fn one() -> Array {
		Array::ones(&[1])
	}
}

/* TODO generic ops for all NDArray<T> */
// impl Not<Array> for Array {
// 	fn not(&self) -> Array
// 	{
// 		let mut p = Array::alloc_with_shape(self.shape());
// 		let arg_it = self._inner.iter();
// 		p._inner.extend(arg_it.map(|&a1|
// 			!a1
// 		));
// 		p
// 	}
// }

// impl BitAnd<Array, Array> for Array {
// 	fn bitand(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::alloc_with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			p._inner.extend(arg_it.map(|(&a1, &a2)|
// 				a1 & a2
// 			));
// 			p
// 		}
// 	}
// }

// impl BitOr<Array, Array> for Array {
// 	fn bitor(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::alloc_with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			p._inner.extend(arg_it.map(|(&a1, &a2)|
// 				a1 | a2
// 			));
// 			p
// 		}
// 	}
// }

// impl BitXor<Array, Array> for Array {
// 	fn bitxor(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::alloc_with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			p._inner.extend(arg_it.map(|(&a1, &a2)|
// 				a1 ^ a2
// 			));
// 			p
// 		}
// 	}
// }

// impl Shl<Array, Array> for Array {
// 	fn shl(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::alloc_with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			p._inner.extend(arg_it.map(|(&a1, &a2)|
// 				a1 << (a2 as uint)
// 			));
// 			p
// 		}
// 	}
// }

// impl Shr<Array, Array> for Array {
// 	fn shr(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::alloc_with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			p._inner.extend(arg_it.map(|(&a1, &a2)|
// 				a1 >> (a2 as uint)
// 			));
// 			p
// 		}
// 	}
// }
