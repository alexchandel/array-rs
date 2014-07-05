use std::iter::Iterator;
use std::vec::Vec;

/// TODO generic NDArray<T> for ints, floats including f128, and bool
#[deriving(Show)]
pub struct Array
{
	_inner: Vec<f64>,	// must never grow!
	_shape: Vec<uint>,	// product equals _inner.len()
	_size: uint			// equals product of _shape and _inner.len()
}

type NDArray = Array; // TODO This typedef is backwards! Make Array = NDArray<T>!

impl Array
{
	pub fn square_demo() -> Array
	{
		let p = Array::with_slice(&[0f64,1f64,2f64,3f64,4f64,5f64,6f64,7f64,8f64]);
		let q = p.reshape(&[3,3]);
		println!("Reshaped!");
		q
	}

	pub fn empty() -> Array
	{
		Array {
			_inner: Vec::new(),
			_shape: vec!(0),
			_size: 0
		}
	}

	pub fn with_scalar(scalar: f64) -> Array
	{
		Array {
			_inner: vec!(scalar),
			_shape: vec!(1),
			_size: 1
		}
	}

	pub fn with_slice(sl: &[f64]) -> Array
	{
		let capacity = sl.len();
		Array {
			_inner: sl.to_owned(),
			_shape: vec!(capacity),
			_size: capacity
		}
	}

	pub fn with_vec(vector: Vec<f64>) -> Array
	{
		let capacity = vector.len();
		Array {
			_inner: vector.clone(),
			_shape: vec!(capacity),
			_size: capacity
		}
	}

	pub fn with_shape(shape: &[uint]) -> Array
	{
		let capacity = shape.iter().fold(1u, |b, &a| a * b);
		Array {
			_inner: Vec::with_capacity(capacity),
			_shape: shape.to_owned(),
			_size: capacity
		}
	}

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
		let prod_f = self._inner.iter().fold(1f64, |b, &a| a * b);
		Array {
			_inner: vec!(prod_f),
			_shape: vec!(1),
			_size: 1
		}
	}

	/// Dot product
	/// TODO multiple dimensions
	pub fn dot(&self, rhs: &Array) -> Array
	{
		//println!("{:}", self._inner);
		let ref v: Vec<f64> = self._inner;
		let ref vr: Vec<f64> = rhs._inner;
		let dot_f = v.iter().zip(vr.iter()).fold(0f64, |b, (&a1, &a2)| b + a1*a2);
		Array {
			_inner: vec!(dot_f),
			_shape: vec!(1),
			_size: 1
		}
	}

	pub fn size(&self) -> uint
	{
		self._shape.iter().fold(1u, |b, &a| a * b)
	}

	pub fn shape<'a>(&'a self) -> &'a [uint]
	{
		self._shape.as_slice()
	}

	/// Return an array with new shape. Fail if numbers of elements differ.
	/// TODO should operate in place?
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
				_shape: shape.to_owned(),
				_size: self._size
			}
		}
		else
		{
			let mut full_shape = shape.to_owned();
			full_shape.push(self._size / capacity);
			Array {
				_inner: self._inner.clone(),
				_shape: full_shape,
				_size: self._size
			}
		}
	}

	pub fn get(&self, index: uint) -> f64
	{
		*self._inner.get(index)
	}

	pub fn set(&mut self, index: uint, value: f64)
	{
		*self._inner.get_mut(index) = value;
	}

	/* TODO serialization and deserialization */

	/* TODO single-element assignment */

	/* TODO sub-array assignment */

	/* TODO sub-array iteration */

	/* TODO broadcasting of fn<f64> like functions */

	/* TODO pretty printing of arrays */
}

/* TODO subarray indexing */
// impl Index<uint, Array> for Array {
// 	fn index(&self, index: &uint) -> Array
// 	{
// 		self._inner.get(index)
// 	}
// }

/// Broadcast addition over all elements. Fail if shapes differ.
/// TODO other ops
impl Add<Array, Array> for Array {
	fn add(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
			{
				*dest = *a1 + *a2;
			}
			p
		}
	}
}

impl Sub<Array, Array> for Array {
	fn sub(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
			{
				*dest = *a1 - *a2;
			}
			p
		}
	}
}

impl Mul<Array, Array> for Array {
	fn mul(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
			{
				*dest = *a1 * *a2;
			}
			p
		}
	}
}

impl Div<Array, Array> for Array {
	fn div(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
			{
				*dest = *a1 / *a2;
			}
			p
		}
	}
}

impl Rem<Array, Array> for Array {
	fn rem(&self, other: &Array) -> Array
	{
		assert!(self._shape == other._shape, "Array shapes must be equal!");
		{
			let mut p = Array::with_shape(self.shape());
			let arg_it = self._inner.iter().zip(other._inner.iter());
			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
			{
				*dest = *a1 + *a2;
			}
			p
		}
	}
}

impl Neg<Array> for Array {
	fn neg(&self) -> Array
	{
		let mut p = Array::with_shape(self.shape());
		let arg_it = self._inner.iter();
		for (dest, a1) in p._inner.mut_iter().zip(arg_it)
		{
			*dest = -*a1;
		}
		p
	}
}

/* TODO generic ops for all NDArray<T> */
// impl Not<Array> for Array {
// 	fn not(&self) -> Array
// 	{
// 		let mut p = Array::with_shape(self.shape());
// 		let arg_it = self._inner.iter();
// 		for (dest, a1) in p._inner.mut_iter().zip(arg_it)
// 		{
// 			*dest = !*a1;
// 		}
// 		p
// 	}
// }

// impl BitAnd<Array, Array> for Array {
// 	fn bitand(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
// 			{
// 				*dest = *a1 & *a2;
// 			}
// 			p
// 		}
// 	}
// }

// impl BitOr<Array, Array> for Array {
// 	fn bitor(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
// 			{
// 				*dest = *a1 | *a2;
// 			}
// 			p
// 		}
// 	}
// }

// impl BitXor<Array, Array> for Array {
// 	fn bitxor(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
// 			{
// 				*dest = *a1 ^ *a2;
// 			}
// 			p
// 		}
// 	}
// }

// impl Shl<Array, Array> for Array {
// 	fn shl(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
// 			{
// 				*dest = *a1 << (*a2 as uint);
// 			}
// 			p
// 		}
// 	}
// }

// impl Shr<Array, Array> for Array {
// 	fn shr(&self, other: &Array) -> Array
// 	{
// 		assert!(self._shape == other._shape, "Array shapes must be equal!");
// 		{
// 			let mut p = Array::with_shape(self.shape());
// 			let arg_it = self._inner.iter().zip(other._inner.iter());
// 			for (dest, (a1, a2)) in p._inner.mut_iter().zip(arg_it)
// 			{
// 				*dest = *a1 >> (*a2 as uint);
// 			}
// 			p
// 		}
// 	}
// }
