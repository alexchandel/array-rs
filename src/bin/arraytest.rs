#![crate_type = "bin"]

extern crate array;

use array::Array;

fn main() {
	let row = Array::with_slice(&[1f64,2f64,3f64]);
	let col = Array::with_slice(&[1f64,10f64,100f64]);
	let d = row.dot(&col);
	println!("{:}", d);

	let sq = Array::square_demo();
	println!("{0},\n{1}", sq, sq.get([0,1]));

	let s = Array::with_slice([1.,0.,0., 0.,2.,0., 0.,0.,3.]).reshape([3,3]);
	let d = s.dot(&s);
	println!("{0}\n{1}", s, d);
}
