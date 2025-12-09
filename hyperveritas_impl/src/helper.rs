#![allow(warnings)]

use ark_ec::pairing::Pairing;
use std::{ops::Deref, primitive, str::FromStr, time::Instant};

use ark_bls12_381::{Bls12_381, Fq, Fr, G1Affine, G2Affine};
use ark_ff::{Field, Fp, Fp2, PrimeField, Zero};
use subroutines::{
    pcs::{
        self,
        prelude::{Commitment, MultilinearKzgPCS, PolynomialCommitmentScheme},
    },
    poly_iop::{
        prelude::{PermutationCheck, ProductCheck, SumCheck, ZeroCheck},
        PolyIOP,
    },
    BatchProof, MultilinearProverParam, PolyIOPErrors,
};

use arithmetic::{eq_eval, merge_polynomials, random_mle_list, VPAuxInfo, VirtualPolynomial};
pub use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_std::{
    end_timer,
    rand::{self, RngCore},
    start_timer,
};
use std::{marker::PhantomData, sync::Arc};
use transcript::IOPTranscript;


use ark_bls12_381::Fr as F;
type PCS = MultilinearKzgPCS<Bls12_381>;
use ark_ff::One;
use ark_std::{rand::RngCore as R, test_rng};
use itertools::Itertools;


pub const irredPolyTable: &[u32] = &[
    0, 0, 7, 11, 19, 37, 67, 131, 285, 529, 1033, 2053, 4179, 8219, 16707, 32771, 69643, 131081,
    262273, 524327, 1048585, 2097157, 4194307, 8388641, 16777351, 33554441, 67108935,
];
pub const flagUseCommits: bool = false;


pub fn vec_to_poly<F: PrimeField>(vec: Vec<F>) -> (Arc<DenseMultilinearExtension<F>>, F) {
    let mut size = (vec.len()).next_power_of_two() - 1;
    let mut nv = 0;
    //We get how much we'll have to pad the image vector.
    while size > 0 {
        size = size >> 1;
        nv += 1;
    }
    //We create the vector that will define our polynomial.
    let mut myVec = Vec::with_capacity(1 << nv);
    let mut sum = F::zero();

    for i in vec.iter() {
        myVec.push(*i);
        sum += i;
    }
    //We do appropriate padding
    for _ in 0..((1 << nv) - vec.len()) {
        myVec.push(F::zero());
    }
    //We convert to a dense multilinear extension, then convert this to a virtual polynomial.
    return (
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, myVec)),
        sum,
    );
}

//This function takes the virtual polynomial coressponding to our matrix, and returns the polynomial on the smaller hypercube F_M(r,b), along with its sum.
pub fn matrix_poly<F: PrimeField>(
    matrix_as_poly: VirtualPolynomial<F>,
    randomness: Vec<F>,
) -> (Arc<DenseMultilinearExtension<F>>, F) {
    let nv = matrix_as_poly.aux_info.num_variables - randomness.len();
    //Let f_M be our matrix as poly. This poly is f_M(r1,..,r_n,b). Where b is hypercube of size nv(M)-nv(r).
    let mut F_M = Vec::with_capacity(1 << nv);
    let mut sum = F::zero();
    let mut evalPtAsVec = Vec::new();
    let mut rand = Vec::new();
    rand.clone_from(&randomness);

    //We anticipate it to be of form F(b,r). Bits are read backwards (1100 corresponds to the number 3).
    let mut profileEvals = 0;
    let mut profileCreatePts = 0;
    for i in 0..1 << nv {
        let now = Instant::now();

        evalPtAsVec.clear();
        rand.clone_from(&randomness);

        for j in 0..(nv) {
            if ((i >> j) & 1 == 1) {
                evalPtAsVec.push(F::one());
            } else {
                evalPtAsVec.push(F::zero());
            }
        }

        evalPtAsVec.append(&mut rand);

        //We now evaluate our polynomial at the correct place.

        //We first create our evaluation point
        profileCreatePts += now.elapsed().as_millis();

        let pt = &evalPtAsVec;
        let now = Instant::now();
        let evalPt = matrix_as_poly.evaluate(pt).unwrap();
        profileEvals += now.elapsed().as_millis();

        F_M.push(evalPt);
        sum += evalPt;
    }
    println!("evaluations took {} seconds.", profileEvals / 1000);
    println!(
        "creating eval pts took {} seconds.",
        profileCreatePts / 1000
    );

    return (
        Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, F_M)),
        sum,
    );
}

//Adapted from web https://users.rust-lang.org/t/how-to-serialize-a-u32-into-byte-array/986
pub fn transform_u32_to_array_of_u8(x: u32) -> [u8; 4] {
    let b1: u8 = ((x >> 24) & 0xff) as u8;
    let b2: u8 = ((x >> 16) & 0xff) as u8;
    let b3: u8 = ((x >> 8) & 0xff) as u8;
    let b4: u8 = (x & 0xff) as u8;
    return [b4, b3, b2, b1];
}

pub fn transform_u128_to_array_of_u8(x: u128) -> [u8; 16] {
    let mut bytes = [0; 16];
    for i in 0..16 {
        bytes[15 - i] = ((x >> 8 * i) & 0xff) as u8;
    }

    return bytes;
}

//This makes a vector size n that is 0,1,2,3,...,255,0,1,2,... all the way up to size.
pub fn makeTestImg(size: usize, cap: usize) -> Vec<F> {
    let mut vec = Vec::new();
    for i in 0..size {
        let mut imod255 = i.clone();
        while imod255 > cap {
            imod255 -= cap
        }
        vec.push(F::from_le_bytes_mod_order(&[imod255.try_into().unwrap()]));
    }
    return vec;
}

pub fn makeTestImgDeleteSoon(size: usize, cap: usize) -> Vec<F> {
    let mut vec: Vec<Fp<ark_ff::MontBackend<ark_bls12_381::FrConfig, 4>, 4>> = Vec::new();
    vec.push(F::from_le_bytes_mod_order(&[1.try_into().unwrap()]));
    vec.push(F::from_le_bytes_mod_order(&[0.try_into().unwrap()]));

    for i in 2..size {
        let mut imod255 = i.clone();
        while imod255 > cap {
            imod255 -= cap
        }
        vec.push(F::from_le_bytes_mod_order(&[imod255.try_into().unwrap()]));
    }
    return vec;
}

pub fn makeConstImg(size: usize, val: usize) -> Vec<F> {
    let mut vec = Vec::new();
    for i in 0..size {
        vec.push(F::from_le_bytes_mod_order(&[val.try_into().unwrap()]));
    }
    return vec;
}

pub fn toFieldVec<F: PrimeField>(u8Vec: &[u8]) -> Vec<F> {
    let mut vec = Vec::new();
    for i in 0..u8Vec.len() {
        vec.push(F::from_le_bytes_mod_order(&[u8Vec[i]]));
    }
    return vec;
}

pub fn toFieldVecFri<F: From<u64>>(u64Vec: &[u64]) -> Vec<F> {
    u64Vec.iter().map(|&x| F::from(x)).collect()
}

pub fn fieldVec<F: From<u64>>(u64Vec: &[u64]) -> Vec<F> {
    u64Vec.iter().map(|&x| F::from(x)).collect()
}

//This is matrix A and vector v, returns h=Av
pub fn matrixMultWithVec<F: PrimeField>(
    numRows: usize,
    numCols: usize,
    A: &[F],
    v: &[F],
) -> Vec<F> {
    let mut h = Vec::new();
    for i in 0..numRows {
        let mut mySum = F::zero();
        for j in 0..numCols {
            mySum += A[i * numCols + j] * v[j];
        }
        h.push(mySum);
    }
    return h;
}

//This is matrix A and vector v, returns h=rA
pub fn vecMultWithMatrix<F: PrimeField>(
    numRows: usize,
    numCols: usize,
    A: &[F],
    r: &[F],
) -> Vec<F> {
    let mut rTa = Vec::new();
    for i in 0..numCols {
        let mut mySum = F::zero();
        for j in 0..numRows {
            mySum += A[i + numCols * j] * r[j];
        }
        rTa.push(mySum);
    }
    return rTa;
}

//MULT A MATRIX AND A SPARSE VECTOR
pub fn matSparseMultVec<F: PrimeField>(
    numRows: usize,
    numCols: usize,
    sprseRep: &[Vec<(usize, F)>],
    r: &[F],
) -> Vec<F> {
    let mut Ar = Vec::new();
    for i in 0..numRows {
        let mut mySum = F::zero();
        for j in 0..sprseRep[i].len() {
            mySum += sprseRep[i][j].1 * r[sprseRep[i][j].0];
        }
        Ar.push(mySum);
    }
    return Ar;
}

pub fn vecMultSparseMat<F: PrimeField>(
    numRows: usize,
    numCols: usize,
    sprseRep: &[Vec<(usize, F)>],
    r: &[F],
) -> Vec<F> {
    let mut Ar = Vec::new();
    for i in 0..numRows {
        let mut mySum = F::zero();
        for j in 0..sprseRep[i].len() {
            mySum += sprseRep[i][j].1 * r[sprseRep[i][j].0];
        }
        Ar.push(mySum);
    }
    return Ar;
}

pub fn main(){
    
}