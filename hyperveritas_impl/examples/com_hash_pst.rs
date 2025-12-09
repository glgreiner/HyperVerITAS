#![allow(warnings)]
use std::ops::Add;
use ark_ec::pairing::Pairing;
use std::fmt::Debug;
use std::ops::Mul;
use ark_ff::MontBackend;
use ark_bls12_381::FrConfig;
use std::ops::AddAssign;

use core::num;
use std::{ops::Deref, primitive, str::FromStr, time::Instant};

use ark_bls12_381::{Bls12_381, Fq, Fr, G1Affine, G2Affine};
use ark_ff::{Field, Fp, Fp2, PrimeField, UniformRand, Zero};
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
use proc_status::ProcStatus;
use std::{marker::PhantomData, sync::Arc};
use transcript::IOPTranscript;
use std::env;

use ark_bls12_381::Fr as F;
type PCS = MultilinearKzgPCS<Bls12_381>;
use ark_ff::One;
use ark_std::{rand::RngCore as R, test_rng};
use itertools::Itertools;
use hyperveritas_impl::{helper::*, image::*, prover::*};

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;


fn run_hash_com_pst(input_size: usize) {
    let mut rng = test_rng();

    let fileName = format!("images/Timings{}.json", input_size);
    let srs = PCS::gen_srs_for_testing(&mut rng, input_size).unwrap();
    let (pcs_param, ver_param) = PCS::trim(&srs, None, Some(input_size)).unwrap();
    println!("params generated");

    let origImg = load_image(&fileName);
    let imgPolyR = vec_to_poly(toFieldVec(&origImg.R)).0;
    let imgPolyG = vec_to_poly(toFieldVec(&origImg.G)).0;
    let imgPolyB = vec_to_poly(toFieldVec(&origImg.B)).0;
    println!("setup done\n");

    let commit_start = Instant::now();
    let img_comR = PCS::commit(pcs_param.clone(), &imgPolyR).unwrap();
    let img_comG = PCS::commit(pcs_param.clone(), &imgPolyG).unwrap();
    let img_comB = PCS::commit(pcs_param.clone(), &imgPolyB).unwrap();
    let elapsed_time = commit_start.elapsed();

    println!("PST Commit Time is {:?} seconds", elapsed_time.as_millis() as f64 / 1000 as f64);
}

fn main(){
    let args: Vec<String> = env::args().collect();

    let first_size = args[1].parse::<usize>().unwrap();
    let mut last_size = first_size;
    if args.len() == 3{
        last_size = args[2].parse::<usize>().unwrap();
    }

    for i in first_size..last_size+1 {
        println!("-----------------------------------------------------------------------");
        println!("PCS Hash, PST. Size: 2^{:?}\n", i);
        let _res = run_hash_com_pst(i);
        println!("-----------------------------------------------------------------------");
    }
}