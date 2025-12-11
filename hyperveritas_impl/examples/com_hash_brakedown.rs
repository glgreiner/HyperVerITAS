#![allow(warnings)]

use core::num;
use proc_status::ProcStatus;
use arithmetic::bit_decompose;
use transcript::IOPTranscript;
use std::{marker::PhantomData, sync::Arc, ops::{Range, Deref}, primitive, str::FromStr, time::Instant, env, array, iter};

use ark_ec::pairing::prepare_g1;
use ark_std::{rand::{RngCore as R, rngs::{OsRng, StdRng}, CryptoRng, RngCore, SeedableRng}, test_rng, };

use rand_chacha::ChaCha8Rng;

use hyperveritas_impl::{types::*, helper::*, image::*};

use plonkish_backend::{
    pcs::{
        Evaluation, PolynomialCommitmentScheme,
        multilinear::{MultilinearBrakedown, MultilinearBrakedownCommitment, additive::{batch_open_one, batch_verify_one},},
    },
    poly::{
        Polynomial,
        multilinear::{rotation_eval, MultilinearPolynomial}
    },
    piop::sum_check::{
        evaluate, SumCheck, VirtualPolynomial,
        classic::{ClassicSumCheck, ClassicSumCheckProver, EvaluationsProver}, 
    },
    util::{
        Itertools, 
        hash::Blake2s,
        new_fields::Mersenne127 as F,
        code::{Brakedown, BrakedownSpec3, BrakedownSpec6},
        expression::{CommonPolynomial, Expression, Query, Rotation}, 
        arithmetic::{BatchInvert, BooleanHypercube, Field as myField}, 
        transcript::{Blake2sTranscript, FiatShamirTranscript, FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript, TranscriptWrite},
    },
};


type Pcs = MultilinearBrakedown<F, Blake2s, BrakedownSpec6>;
type VT = FiatShamirTranscript<Blake2s, std::io::Cursor<Vec<u8>>>;


fn run_hash_com_brakedown(input_size: usize) {
    let mut rng = test_rng();

    let length = input_size;
    
    let (pp, vp) = {
        let poly_size = 1 << length;
        let param = Pcs::setup(poly_size, 4, &mut rng).unwrap();
        Pcs::trim(&param, poly_size, 4).unwrap()
    };


    let mut transcript = Blake2sTranscript::new(());

    let fileName = format!("images/Timings{}.json", input_size);
    let origImg = load_image(&fileName);


    let mut rgb_evals =
        [fieldVec::<F>(&origImg.R.iter().map(|&x| x as u64).collect::<Vec<_>>()),
         fieldVec::<F>(&origImg.G.iter().map(|&x| x as u64).collect::<Vec<_>>()),
         fieldVec::<F>(&origImg.B.iter().map(|&x| x as u64).collect::<Vec<_>>()),];

    // first create the proper multilinears
    let mut img_polys: Vec<MultilinearPolynomial<F>> = Vec::new();

    img_polys.push(MultilinearPolynomial::<F>::new(rgb_evals[0].clone()));
    img_polys.push(MultilinearPolynomial::<F>::new(rgb_evals[1].clone()));
    img_polys.push(MultilinearPolynomial::<F>::new(rgb_evals[2].clone()));

    let commit_start = Instant::now();
    let imgComs = Pcs::batch_commit_and_write(&pp, &img_polys, &mut transcript);
    let elapsed_time = commit_start.elapsed();

    println!("Brakedown 127 Commit Time is {:?} seconds", elapsed_time.as_millis() as f64 / 1000 as f64);

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
        println!("PCS Hash, Brakedown 127. Size: 2^{:?}\n", i);
        let _res = run_hash_com_brakedown(i);
        println!("-----------------------------------------------------------------------");
    }
}