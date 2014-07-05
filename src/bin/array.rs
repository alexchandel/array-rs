#![crate_id = "array"]
#![crate_type = "bin"]

extern crate array;

use array::Array;

fn main() {
	let row = Array::with_slice([1f64,2f64,3f64]);
	let col = Array::with_slice([1f64,10f64,100f64]);
	let d = row.dot(&col);
	println!("{:}", d);

	let sq = Array::square_demo();
	println!("{:}", sq);
}
