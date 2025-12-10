#![allow(warnings)]

use ark_bls12_381::{Bls12_381, Fr};

use ark_gemini::absorbcircuit::{AbsorbCircuit, poseidon_parameters_for_test};

use ark_poly::univariate::DensePolynomial;
use ark_poly::{Polynomial};
use ark_std::test_rng;
use ark_std::UniformRand;
use ark_poly::domain::EvaluationDomain;
use ark_poly::domain::general::GeneralEvaluationDomain;
use ark_poly::evaluations::univariate::Evaluations;
use ark_gemini::kzg::CommitterKey;
use ark_gemini::kzg::VerifierKey;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use ark_poly::DenseUVPolynomial;
use ark_std::Zero;
use ark_ff::fields::Field;
use ark_gemini::kzg::Commitment;
use ark_gemini::kzg::EvaluationProof;
use std::fs::{File, read, OpenOptions};
use std::io::{BufRead, BufReader, Write, Result};
use ark_ff::BigInteger;
use sha256::try_digest;
use sha256::digest;
use std::path::Path;
use std::thread;
use std::env;

use ark_std::rand::{RngCore, SeedableRng};
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_crypto_primitives::sponge::{CryptographicSponge, FieldBasedCryptographicSponge};

use ark_bls12_381::{G1Affine as GAffine};
use ark_ff::PrimeField;

use ark_groth16::Groth16;
use ark_crypto_primitives::snark::{CircuitSpecificSetupSNARK, SNARK};
use ark_groth16::ProvingKey;
use ark_groth16::VerifyingKey;
use ark_groth16::Proof;

use num_bigint::BigUint;
use proc_status::ProcStatus;

mod threeGray;
use threeGray::*;

mod oneHash;
use oneHash::*;

static name : &str = "fullSystemGray_Hash";

static EXPONENT : u32 = 8;
static PIXEL_RANGE : i32 = 2_i32.pow(EXPONENT);
static HASH_LENGTH : usize = 128;

fn print_time_since(start: u128, last: u128, tag: &str, size: &usize, chan: &str) -> u128 {
    let now = SystemTime::now();
    let now_epoc = now
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let now = now_epoc.as_millis();

    let path = format!("output/{}_{}_{}.txt", name.to_string(), size, chan.to_string());

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path).unwrap();

    writeln!(file, "{:?}; time since start {:?}; time since last check: {:?} seconds", tag, (now - start) as f64 / 1000 as f64, (now - last) as f64 / 1000 as f64);
    return now;
}

// ConvertVecToPoly (Section 5.1)
fn interpolate(vals : Vec<Fr>, domain: GeneralEvaluationDomain::<Fr>) -> DensePolynomial<Fr> {
    let evaluations = Evaluations::<Fr, GeneralEvaluationDomain<Fr>>::from_vec_and_domain(vals, domain);
    return evaluations.interpolate();
}

fn get_filename(prefix: &str, postfix: &str, size: &usize) -> String {
    let mut filename = prefix.to_owned();
    filename.push_str(&size.to_string());
    filename.push_str(postfix);
    filename.push_str(".txt");
    return filename
}

fn read_photo(prefix: &str,  postfix: &str, size: &usize) -> BufReader<File> {
    let file = File::open(get_filename(prefix, postfix, size)).expect("Unable to open file");
    return BufReader::new(file);
}

// gets sha256 of commitments for random challenges
fn get_sha256_of_commitments(commitments: Vec<Commitment<Bls12_381>>, instance_hash: &str, num_elements: usize) -> Vec<Fr> {
    let mut byte_vec = Vec::new();
    for commitment in commitments {
        let affine_rep = GAffine::from(commitment.0);
        let mut bytes_x1 = affine_rep.x.into_bigint().to_bytes_le();
        let mut bytes_y1 = affine_rep.y.into_bigint().to_bytes_le();

        byte_vec.append(&mut bytes_x1);
        byte_vec.append(&mut bytes_y1);
        
    }

    let s = format!("{:?}{:?}", &byte_vec, instance_hash);
    let mut val = digest(s);

    let mut ret = Vec::new();

    for _ in 0..num_elements/2 {
        let sha2561 = u128::from_str_radix(&val[0..32], 16).unwrap();
        ret.push(Fr::from(sha2561));
        let sha2562 = u128::from_str_radix(&val[32..64], 16).unwrap();
        ret.push(Fr::from(sha2562));
        val = digest(val);
    }
    
    return ret;
}

// Performs all prover work (i.e., polynomial calculations, commitments, and openings)
fn eval_polynomials(domain: GeneralEvaluationDomain::<Fr>, start: u128, instance_hash: &str, time_ck: &CommitterKey::<Bls12_381>,  postfix: &str, size: &usize, D: usize)  
-> (Vec<Commitment<Bls12_381>>, Vec<Commitment<Bls12_381>>, Vec<Commitment<Bls12_381>>, Vec<Vec<Fr>>, Vec<Vec<Fr>>, Vec<Vec<Fr>>, Vec<Fr>, EvaluationProof<Bls12_381>, EvaluationProof<Bls12_381>, EvaluationProof<Bls12_381>)  {
    let mut rng = &mut test_rng();

    // polynomials will hold the polynomials we commit to
    let mut polynomials0 = Vec::new();

    // w_vals = [0, 1,...,PIXEL_RANGE - 1]
    let mut w_vals = Vec::new();
    for i in 0..PIXEL_RANGE {
        let i_in_fr = Fr::from(i);
        w_vals.push(i_in_fr);
    }

    // w[X] = poly(w_vals)
    let w = interpolate(w_vals, domain);
    // println!("w done");
    polynomials0.push(w.coeffs.clone());

    // v_vals = [pixel_0,...,pixel_{D-1}]
    let mut v_vals = Vec::new();
    // z_vals = [sort(v || w)]
    let mut z_vals : Vec<Fr> = Vec::new();

    // reading in photo pixels...
    let file = read_photo("./images/Veri", postfix, size);
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<i32>().unwrap();

        let v_point = Fr::from(i as i32); 
        v_vals.push(v_point);
        z_vals.push(v_point);

    }

    // v[X] = poly(v_vals)
    let v = interpolate(v_vals, domain);
    // println!("v done");

    polynomials0.push(v.coeffs.clone());

    for i in 0..PIXEL_RANGE {
        let i_in_fr = Fr::from(i);
        z_vals.push(i_in_fr);
    }
    // pad z_vals so that [z[omega*x] - z[x][1 - (z[omega*x] - z[x])] = 0 still holds true
    let z_vals_length = z_vals.len();
    for _ in 0..domain.size() - z_vals_length {
        z_vals.push(Fr::from(PIXEL_RANGE - 1));
    }
    z_vals.sort();

    // z[X] = poly(z_vals)
    let z = interpolate(z_vals.clone(), domain);
    // println!("z prods done");
    polynomials0.push(z.coeffs.clone());

    let time_batched_commitments0 = time_ck.batch_commit(&polynomials0);

    // permutation challenge
    let gamma = get_sha256_of_commitments(time_batched_commitments0.clone(), instance_hash, 2)[0];

    // Permutation argument
    // We want to prove:
    //           product_{i=0}^{D-1}(v_i + gamma) * product_{i=0}^{PIXEL_RANGE-1}(w_i + gamma) = product_{i=0}^{D + PIXEL_RANGE - 1}(z_i + gamma) 
    // where v holds the image pixels, w is the range that the pixel values must lie in [0, PIXEL_RANGE-1],
    // and z is the sorted concatentation of v and w

    let mut polynomials = Vec::new();

    // w_prod_vals = [1, (gamma), [(gamma)(1 + gamma)],...,[(gamma)...(PIXEL_RANGE - 1 + gamma)]]
    let mut w_prod_vals = Vec::new();
    let mut product = Fr::from(1u64);
    w_prod_vals.push(product);

    for i in 0..PIXEL_RANGE {
        let i_in_fr = Fr::from(i);
        product *= i_in_fr + gamma;
        w_prod_vals.push(product);
    }

    let w_prod_vals_len = w_prod_vals.len();
    for _ in 0..domain.size() - w_prod_vals_len {
        product *= gamma;
        w_prod_vals.push(product);
    }

    // w_prod_omega_vals = [(gamma), [(gamma)(1 + gamma)],...,[(gamma)...(PIXEL_RANGE + gamma)], 1]
    let mut w_prod_omega_vals = Vec::new();
    for i in 1..w_prod_vals.len() {
        w_prod_omega_vals.push(w_prod_vals[i]);
    }
    w_prod_omega_vals.push(w_prod_vals[0]);

    // for all i \in [1, PIXEL_RANGE + 1], w_prod[omega^i] = \prod_{j=0}^{i-1}(w_j + gamma)
    let w_prod = interpolate(w_prod_vals, domain);

    // w_prod_omega[X] = w_prod[omega*X]
    let w_prod_omega = interpolate(w_prod_omega_vals, domain);
    // println!("w prods done");

    // n_1[X] = omega^{|domain| - 1} - X
    // We use n_1[X] to ensure that the permutation check equation holds for omega^{|domain} -1}
    let mut n_1_coeffs = Vec::new();
    n_1_coeffs.push(domain.group_gen().pow(&[(domain.size() - 1) as u64]));
    n_1_coeffs.push(Fr::from(-1));
    let n_1 = DensePolynomial::<Fr>::from_coefficients_vec(n_1_coeffs);

    let mut gamma_coeffs = Vec::new();
    gamma_coeffs.push(gamma);
    let gamma_poly = DensePolynomial::<Fr>::from_coefficients_vec(gamma_coeffs);

    // q_w[X] = (w_prod[omega * X] - (w_prod[X] * (gamma + w[X]))) * n_1[X] / Z_H[X]
    let (q_w, r_w) = (&(&w_prod_omega - &(&w_prod * &(&gamma_poly + &w))) * &n_1).divide_by_vanishing_poly(domain).unwrap();
    assert!(r_w.is_zero());

    // println!("q_w done");

    // Will commit to w[X], w_prod[X], q_w[X]
    polynomials.push(w_prod.coeffs.clone());
    polynomials.push(q_w.coeffs.clone());

    // v_prod_vals = [1, (pixel_0 + gamma), [(pixel_0 + gamma)(pixel_1 + gamma)],...,[(pixel_0 + gamma)...(pixel_{D-1} + gamma)]]
    let mut v_prod_vals = Vec::new();
    let mut product = Fr::from(1u64);
    v_prod_vals.push(product);

    // reading in photo pixels...
    let file = read_photo("./images/Veri", postfix, size);
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<i32>().unwrap();

        let v_point = Fr::from(i as i32); 

        product *= v_point + gamma;
        v_prod_vals.push(product);
    }

    for _ in 0..domain.size() - D - 1 {
        product *= gamma;
        v_prod_vals.push(product);
    }

    // v_prod_omega_vals = [(pixel_0 + gamma), [(pixel_0 + gamma)(pixel_1 + gamma)],...,[(pixel_0 + gamma)...(pixel_{D-1} + gamma)], 1]
    let mut v_prod_omega_vals = Vec::new();
    for i in 1..v_prod_vals.len() {
        v_prod_omega_vals.push(v_prod_vals[i]);
    }
    v_prod_omega_vals.push(v_prod_vals[0]);

    // for all i \in [1, D + 1], v_prod[omega^i] = \prod_{j=0}^{i-1}(v_j + gamma)
    let v_prod = interpolate(v_prod_vals, domain);

    // v_prod_omega[X] = v_prod[omega*X]
    let v_prod_omega = interpolate(v_prod_omega_vals, domain);
    // println!("v prods done");

    // q_v[X] = (v_prod[omega * X] - (v_prod[X] * (gamma + v[X]))) * n_1[X] / Z_H[X]
    let (q_v, r_v) = (&(&v_prod_omega - &(&v_prod * &(&gamma_poly + &v))) * &n_1).divide_by_vanishing_poly(domain).unwrap();
    assert!(r_v.is_zero());
    // println!("r_v prods done");

    // Will commit to v[X], v_prod[X], q_v[X]
    polynomials.push(v_prod.coeffs.clone());
    polynomials.push(q_v.coeffs.clone());

    // z_prod_vals = [1, z_vals_0 + gamma, [(z_0 + gamma)(z_vals_1 + gamma)],...,[(z_vals_0 + gamma)...(z_vals_{PIXEL_RANGE + D - 1} + gamma)]]
    let mut z_prod_vals = Vec::new();
    let mut product = Fr::from(1u64);
    z_prod_vals.push(product);
    for i in 0..z_vals.len() - 1 {
        product *= z_vals[i] + gamma;
        z_prod_vals.push(product);
    }

    // Range argument
    // We want to prove for the z constructed above that:
    //      (z[X] - z[omega*X])(1 - (z[X] - z[omega*X]) = 0 mod Z_H[X]

    // z_omega_vals = [z_vals_0 + gamma,...,[(z_vals_0 + gamma)...(z_vals_{PIXEL_RANGE + D - 1} + gamma)], 1]
    let mut z_omega_vals = Vec::new();
    for i in 1..z_vals.len() {
        z_omega_vals.push(z_vals[i]);
    }
    z_omega_vals.push(z_vals[0]);

    // z_prod_omega_vals = [z_vals_0 + gamma, [(z_vals_0 + gamma)(z_vals_1 + gamma)],...,[(z_vals_0 + gamma)...(z_vals_{PIXEL_RANGE + D - 1} + gamma)], 1]
    let mut z_prod_omega_vals = Vec::new();
    for i in 1..z_prod_vals.len() {
        z_prod_omega_vals.push(z_prod_vals[i]);
    }
    z_prod_omega_vals.push(z_prod_vals[0]);

    // for all i \in [1, PIXEL_RANGE + D], z_prod[omega^i] = \prod_{j=0}^{i-1}(z_j + gamma)
    let z_prod = interpolate(z_prod_vals, domain);

    // z_prod_omega[X] = z_prod[omega*X]
    let z_prod_omega = interpolate(z_prod_omega_vals, domain);
    // println!("z_omega prods done");

    // q_v[X] = (v_prod[omega * X] - (v_prod[X] * (gamma + v[X]))) * n_1[X] / Z_H[X]
    let (q_z, r_z) = (&(&z_prod_omega - &(&z_prod * &(&gamma_poly + &z))) * &n_1).divide_by_vanishing_poly(domain).unwrap();
    assert!(r_z.is_zero());
    // println!("q_z prods done");

    // z_omega[X] = z[omega*X]
    let z_omega = interpolate(z_omega_vals, domain);

    let mut one_coeffs = Vec::new();
    one_coeffs.push(Fr::from(1));
    
    let one = DensePolynomial::<Fr>::from_coefficients_vec(one_coeffs);
 
    // q_range[X] = (z[X] - z[omega*X])(1 - (z[X] - z[omega*X]) * n_1[X] / Z_H[X]
    let (q_range, r_range) = (&(&(&z_omega - &z) * &(&one - &(&z_omega - &z))) * &n_1).divide_by_vanishing_poly(domain).unwrap();

    assert!(r_range.is_zero());
    // println!("r_range prods done");

    // Will commit to z[X], z_prod[X], q_z[X], q_range[X]
    polynomials.push(z_prod.coeffs.clone());
    polynomials.push(q_z.coeffs.clone());
    polynomials.push(q_range.coeffs.clone());

    // We commit in batches for memory reasons
    let time_batched_commitments1 = time_ck.batch_commit(&polynomials);
    // println!("first commitment done");

    // Now we prove knowledge of actual hash value (Section 5.5) 
    // Want to generate a[X] and prove that Equation 11 in Section 5.5 holds for
    // this a[X] and the v[X] generated above

    // Use commitments to generate random coefficients [r_0,...,r_{HASH_LENGTH-1}]
    // for random linear combination of sum checks
    let hash_coeffs = get_sha256_of_commitments(time_batched_commitments1.clone(), "", HASH_LENGTH);

    let mut rng = &mut test_rng();

    // Let A be the public hashing matrix (we will generate it with a PRG)
    // a_vals = [\sum_{i=0}{HASH_LENGTH-1}r_i * A_{i, 0},...,\sum_{i=0}{HASH_LENGTH-1}r_i * A_{i, D - 1}]
    let mut a_vals = Vec::new();

    // h_sum_vals = [0, v_vals_0 * a_vals_0 ,..., \sum_{i=0}^{D - 1} v_vals_0 * a_vals_0]
    let mut h_sum_vals = Vec::new();

    // h_sum_omega_vals = [\sum_{i=0}^{1} v_vals_0 * a_vals_0,...,\sum_{i=0}^{D - 1} v_vals_0 * a_vals_0, v_vals_0 * a_vals_0]
    let mut h_sum_omega_vals = Vec::new();
    h_sum_vals.push(Fr::from(0u64));
    let mut sum = Fr::from(0u64);

    // Re-read in pixels
    let file = read_photo("./images/Veri", postfix, size);
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<i32>().unwrap();

        let v_point = Fr::from(i as i32); 

        let mut a_point = Fr::from(0); 
        for j in 0..hash_coeffs.len() {
            a_point += Fr::rand(rng) * hash_coeffs[j];
        }
        a_vals.push(a_point);

        sum += v_point * a_point;
        h_sum_vals.push(sum);
        h_sum_omega_vals.push(sum);
    }

    for _ in 0..domain.size() - D - 1 {
        h_sum_vals.push(sum);
        h_sum_omega_vals.push(sum);
    }
    h_sum_omega_vals.push(Fr::from(0u64));


    // for all i \in [0, D - 1], a[omega^i] = \sum_{j=0}{HASH_LENGTH-1}r_j * A_{j, i}
    let a = interpolate(a_vals, domain); 

    // for all i \in [0, D], h_sum[omega^i] = \sum_{j=0}^{i} v_vals_j * a_vals_j
    let h_sum = interpolate(h_sum_vals, domain);

    // h_sum_omega[X] = h_sum[omega*X]
    let h_sum_omega = interpolate(h_sum_omega_vals, domain);
    // println!("h_sum_omega prods done");

    // q_h_sum[X] = (h_sum[omega*X] - h_sum[X] - (v[X] * a[X]))* n_1[X] / Z_H[X]
    let (q_h_sum, r_h_sum) = (&(&(&h_sum_omega - &h_sum) - &(&v * &a))* &n_1).divide_by_vanishing_poly(domain).unwrap();
    assert!(r_h_sum.is_zero());
    // println!("q_h_sum prods done");

    // Second set of polynomials we commit to
    let mut polynomials2 = Vec::new();
    let mut evals2 = Vec::new();

    // Will commit to a[X], h_sum[X], q_h_sum[X]
    polynomials2.push(a.coeffs.clone());
    polynomials2.push(h_sum.coeffs.clone());
    polynomials2.push(q_h_sum.coeffs.clone());

    // Commit
    let time_batched_commitments2 = time_ck.batch_commit(&polynomials2);
    // println!("second commitment done");

    // PRODUCE OPENING PROOFS

    // alpha is random challenge that we get by hashing commitments (i.e., we use Fiat-Shamir)
    // eta1 and eta2 are the challenges we use to batch evaluation proofs
    let hashes = get_sha256_of_commitments(time_batched_commitments2.clone(), "", 4);
    let alpha = hashes[0];
    let eta0 = hashes[1];
    let eta1 = hashes[2];
    let eta2 = hashes[3];

    // We batch open all committed polynomials at alpha, omega*alpha D, PIXEL_RANGE, D + PIXEL_RANGE
    let mut eval_points = Vec::new();
    eval_points.push(alpha);
    eval_points.push(domain.group_gen() * alpha);
    eval_points.push(domain.group_gen().pow(&[(D) as u64]));
    eval_points.push(domain.group_gen().pow(&[(PIXEL_RANGE) as u64]));
    eval_points.push(domain.group_gen().pow(&[(D + PIXEL_RANGE as usize) as u64]));

    // Evaluate zeroth set of batched polynomials
    // evals0 will hold the evaluations of the polynomials
    let mut evals0 = Vec::new();

    let mut w_evals = Vec::new();
    for x in eval_points.iter() {
        w_evals.push(w.evaluate(x));
    }
    evals0.push(w_evals);
     let mut v_evals = Vec::new();
    for x in eval_points.iter() {
        v_evals.push(v.evaluate(x));
    }
    evals0.push(v_evals);
    let mut z_evals = Vec::new();
    for x in eval_points.iter() {
        z_evals.push(z.evaluate(x));
    }
    evals0.push(z_evals);

    let proof0 = time_ck.batch_open_multi_points(
        &polynomials0.iter().collect::<Vec<_>>(),
        &eval_points,
        &eta0,
    );
    // println!("zeroth proof done");

    // Evaluate first set of batched polynomials
    // evals will hold the evaluations of the polynomials
    let mut evals1 = Vec::new();

    let mut w_prod_evals = Vec::new();
    for x in eval_points.iter() {
        w_prod_evals.push(w_prod.evaluate(x));
    }
    evals1.push(w_prod_evals);

    let mut q_w_evals = Vec::new();
    for x in eval_points.iter() {
        q_w_evals.push(q_w.evaluate(x));
    }
    evals1.push(q_w_evals);

    let mut v_prod_evals = Vec::new();
    for x in eval_points.iter() {
        v_prod_evals.push(v_prod.evaluate(x));
    }
    
    evals1.push(v_prod_evals);

    let mut q_v_evals = Vec::new();
    for x in eval_points.iter() {
        q_v_evals.push(q_v.evaluate(x));
    }
    evals1.push(q_v_evals);

    let mut z_prod_evals = Vec::new();
    for x in eval_points.iter() {
        z_prod_evals.push(z_prod.evaluate(x));
    }
    evals1.push(z_prod_evals);

    let mut q_z_evals = Vec::new();
    for x in eval_points.iter() {
        q_z_evals.push(q_z.evaluate(x));
    }   
    evals1.push(q_z_evals);

    let mut q_range_evals = Vec::new();
    for x in eval_points.iter() {
        q_range_evals.push(q_range.evaluate(x));
    }   
    evals1.push(q_range_evals);

    // Produce opening proofs for first set of batched commitments
    let proof1 = time_ck.batch_open_multi_points(
        &polynomials.iter().collect::<Vec<_>>(),
        &eval_points,
        &eta1,
    );
    // println!("first proof done");

    // Evaluate second set of batched polynomials
    let mut a_evals = Vec::new();
    for x in eval_points.iter() {
        a_evals.push(a.evaluate(x));
    }
    evals2.push(a_evals);

    let mut h_sum_evals = Vec::new();
    for x in eval_points.iter() {
        h_sum_evals.push(h_sum.evaluate(x));
    }
    evals2.push(h_sum_evals);

    let mut q_h_sum_evals = Vec::new();
    for x in eval_points.iter() {
        q_h_sum_evals.push(q_h_sum.evaluate(x));
    }
    evals2.push(q_h_sum_evals);

    // Produce opening proofs for second set of batched commitments
    // let eta2: Fr = u128::rand(&mut rng).into();
    let proof2 = time_ck.batch_open_multi_points(
        &polynomials2.iter().collect::<Vec<_>>(),
        &eval_points,
        &eta2,
    );
    // println!("second proof done");
    

    return (time_batched_commitments0, time_batched_commitments1, time_batched_commitments2, evals0, evals1, evals2, hash_coeffs, proof0, proof1, proof2);
}



fn get_hash_proof_size(time_batched_commitments0: &Vec<Commitment<Bls12_381>>, time_batched_commitments1: &Vec<Commitment<Bls12_381>>, time_batched_commitments2: &Vec<Commitment<Bls12_381>>, 
                        evals0: &Vec<Vec<Fr>>, evals1: &Vec<Vec<Fr>>, evals2: &Vec<Vec<Fr>>, hash_coeffs: &Vec<Fr>, 
                        proof0: &EvaluationProof<Bls12_381>, proof1: &EvaluationProof<Bls12_381>, proof2: &EvaluationProof<Bls12_381>) -> (usize, usize) {

    let mut total_bls12_381_elems = 0;

    let mut total_256_bit_elems = 0;

    // commits are each one group elem, so add the number of commits to the total
    total_bls12_381_elems += time_batched_commitments0.len();
    total_bls12_381_elems += time_batched_commitments1.len();
    total_bls12_381_elems += time_batched_commitments2.len();

    // add the elements in the evals0 vectors
    for eval_vec in evals0 {
        total_256_bit_elems += eval_vec.len();
    }

    // add the elements in the evals1 vectors
    for eval_vec in evals1 {
        total_256_bit_elems += eval_vec.len();
    }

    // add the elements in the evals2 vectors
    for eval_vec in evals2 {
        total_256_bit_elems += eval_vec.len();
    }

    // add the hash elems
    total_256_bit_elems += hash_coeffs.len();

    // count the group elems from the three EvaluationProofs (proof0, proof1, proof2)
    total_bls12_381_elems += 3;


    return (total_bls12_381_elems, total_256_bit_elems) 
}

fn run_full_gray(size: &usize) {
    println!("Full System (Hash + Grayscale), VerITAS KZG. Size: 2^{:?}\n", size);
    let D = 1 << size;

    // SETUP 
    let prover_setup_start = Instant::now();

    let (r_domain, r_instance_hash, r_time_ck) = setup(size, D, "R");
    let r_time_vk = VerifierKey::from(&r_time_ck);
    let r_instance_hash_str = r_instance_hash.clone();

    let (g_domain, g_instance_hash, g_time_ck) = setup(size, D, "G");
    let g_time_vk = VerifierKey::from(&g_time_ck);
    let g_instance_hash_str = g_instance_hash.clone();

    let (b_domain, b_instance_hash, b_time_ck) = setup(size, D, "B");
    let b_time_vk = VerifierKey::from(&b_time_ck);
    let b_instance_hash_str = b_instance_hash.clone();

    let elapsed_time_setup = prover_setup_start.elapsed().as_millis();
    println!("## Total setup time: {:?} seconds\n", elapsed_time_setup as f64 / 1000 as f64);

    // Hash Pre Image
    let prover_hash_start = Instant::now();

    // R channel hash proof
    let r_D = D+0;
    let r_size = size+0;
    let thread_hash_proof_R= thread::spawn( move || {
        // setup timing stuff
        let start = SystemTime::now();
        let start_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let start = start_epoch.as_millis();
        let mut last = start;

        println!("R Thread Running");

        // PROVER
        let prover_out_R = eval_polynomials(r_domain.clone(), start, r_instance_hash_str.as_str(), &r_time_ck, "R", &r_size, r_D);

        println!("R Thread Done");

        prover_out_R
    });

    // G channel hash proof
    let g_D = D+0;
    let g_size = size+0;
    let thread_hash_proof_G = thread::spawn( move || {
        // setup timing stuff
        let start = SystemTime::now();
        let start_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let start = start_epoch.as_millis();
        let mut last = start;

        println!("G Thread Running");

        // PROVER
        let prover_out_G = eval_polynomials(g_domain.clone(), start, g_instance_hash_str.as_str(), &g_time_ck, "G", &g_size, g_D);

        println!("G Thread Done");

        prover_out_G     
    });

    // B channel hash proof
    let b_D = D+0;
    let b_size = size+0;
    let thread_hash_proof_B = thread::spawn( move || {
        // setup timing stuff
        let start = SystemTime::now();
        let start_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let start = start_epoch.as_millis();
        let mut last = start;

        println!("B Thread Running");

        // PROVER
        let prover_out_B = eval_polynomials(b_domain.clone(), start, b_instance_hash_str.as_str(), &b_time_ck, "B", &b_size, b_D);

        println!("B Thread Done");

        prover_out_B          
    });

    // obtain the proof stuff from the threads
    let prover_out_R = thread_hash_proof_R.join().unwrap();
    let prover_out_G = thread_hash_proof_G.join().unwrap();
    let prover_out_B = thread_hash_proof_B.join().unwrap();

    let elapsed_time_prover_hash = prover_hash_start.elapsed();
    println!("\n## Prover Runtime for RGB Hash Proofs: {:?} seconds\n", elapsed_time_prover_hash.as_millis() as f64 / 1000 as f64);


    // NOW Image Transformation.
    
    // Grayscale SETUP

    let prover_setup_start = Instant::now();

    let grayscale_setup = threeGray::grayscale_system_setup(size);

    let elapsed_time_setup = prover_setup_start.elapsed().as_millis();
    println!("Total setup time: {:?} seconds", elapsed_time_setup as f64 / 1000 as f64);

    // NOW PROVE Grayscale

    let prover_grayscale_start = Instant::now();
    
    let grayscale_proof = threeGray::grayscale_system_prove(&size, grayscale_setup.2, grayscale_setup.3, grayscale_setup.4,
                                                                                                               grayscale_setup.5, grayscale_setup.6, grayscale_setup.7,
                                                                                                               grayscale_setup.8, grayscale_setup.9);

    let elapsed_time_prover = prover_grayscale_start.elapsed();
    println!("\n## Prover Runtime for Grayscale Proof: {:?} seconds", elapsed_time_prover.as_millis() as f64 / 1000 as f64);
    println!("\n## Total Prover Runtime (hash + grayscale): {:?} seconds", (elapsed_time_prover.as_millis()+elapsed_time_prover_hash.as_millis()) as f64 / 1000 as f64);
    println!("-----------------------------------------------------------------------");
    println!("Computing Proof Sizes");

    let (  red_hash_bls_elems,   red_hash_256_elems) = get_hash_proof_size(&prover_out_R.0, &prover_out_R.1, &prover_out_R.2, 
                                                                                         &prover_out_R.3, &prover_out_R.4, &prover_out_R.5, 
                                                                                         &prover_out_R.6, 
                                                                                         &prover_out_R.7, &prover_out_R.8, &prover_out_R.9);

    let (green_hash_bls_elems, green_hash_256_elems) = get_hash_proof_size(&prover_out_G.0, &prover_out_G.1, &prover_out_G.2, 
                                                                                         &prover_out_G.3, &prover_out_G.4, &prover_out_G.5, 
                                                                                         &prover_out_G.6, 
                                                                                         &prover_out_G.7, &prover_out_G.8, &prover_out_G.9);

    let ( blue_hash_bls_elems,  blue_hash_256_elems) = get_hash_proof_size(&prover_out_B.0, &prover_out_B.1, &prover_out_B.2, 
                                                                                         &prover_out_B.3, &prover_out_B.4, &prover_out_B.5, 
                                                                                         &prover_out_B.6, 
                                                                                         &prover_out_B.7, &prover_out_B.8, &prover_out_B.9);

    println!("r Hash Proof: {:?} BLS12_381, {:?} 256 bit elems",   red_hash_bls_elems,   red_hash_256_elems);
    println!("g Hash Proof: {:?} BLS12_381, {:?} 256 bit elems", green_hash_bls_elems, green_hash_256_elems);
    println!("b Hash Proof: {:?} BLS12_381, {:?} 256 bit elems",  blue_hash_bls_elems,  blue_hash_256_elems);
    
    println!("");

    let total_hash_bls_elems = red_hash_bls_elems + green_hash_bls_elems + blue_hash_bls_elems;
    let total_hash_256_elems = red_hash_256_elems+ green_hash_256_elems + blue_hash_256_elems;

    println!("total elems : {:?} BLS12_381, {:?} 256 bit elems", total_hash_bls_elems, total_hash_256_elems);

    let total_hash_bytes = total_hash_bls_elems * 48 + total_hash_256_elems * 32;

    println!("## total hash proof bytes : {:?} Bytes", total_hash_bytes);

    println!("\n");
    
    // now we get the size of the grayscale proof

    // get size of compressed proofs (this includes public inputs)
    let grayscale_proof_bytes_vec = grayscale_proof.1.to_bytes();
    let mut grayscale_proof_bytes = grayscale_proof_bytes_vec.len();

    // calculate size of public inputs
    let grayscale_public_field_elems = grayscale_proof.1.public_inputs.len();
    let grayscale_public_field_elems_bytes = grayscale_public_field_elems * 8;

    // subtract public input size
    grayscale_proof_bytes = grayscale_proof_bytes - grayscale_public_field_elems_bytes;

    println!("## total grayscale proof size: {:?} Bytes", grayscale_proof_bytes);

    let total_proof_size = total_hash_bytes + grayscale_proof_bytes;

    println!("\n## TOTAL PROOF SIZE (hash + grayscale): {:?} Bytes", total_proof_size);


    println!("-----------------------------------------------------------------------");

    println!("Starting Verifier now");

    let verifier_start = Instant::now();

    // verifying grayscale proof

    threeGray::grayscale_system_verify(grayscale_proof.0, grayscale_proof.1, grayscale_setup.0, grayscale_setup.1, size);

    // verifying hash proofs

    oneHash::verify(prover_out_R.0, prover_out_R.1, prover_out_R.2, 
                       prover_out_R.3, prover_out_R.4, prover_out_R.5, 
                       prover_out_R.6, 
                       prover_out_R.7, prover_out_R.8, prover_out_R.9,
                       r_instance_hash.as_str(), D, r_domain, r_time_vk);

    oneHash::verify(prover_out_G.0, prover_out_G.1, prover_out_G.2, 
                       prover_out_G.3, prover_out_G.4, prover_out_G.5, 
                       prover_out_G.6, 
                       prover_out_G.7, prover_out_G.8, prover_out_G.9,
                       g_instance_hash.as_str(), D, g_domain, g_time_vk);

    oneHash::verify(prover_out_B.0, prover_out_B.1, prover_out_B.2, 
                       prover_out_B.3, prover_out_B.4, prover_out_B.5, 
                       prover_out_B.6, 
                       prover_out_B.7, prover_out_B.8, prover_out_B.9,
                       b_instance_hash.as_str(), D, b_domain, b_time_vk);

    let elapsed_time_verifier = verifier_start.elapsed();
    println!("Verifier Runtime: {:?} seconds", elapsed_time_verifier.as_millis() as f64 / 1000 as f64);
    println!("-----------------------------------------------------------------------");
}

fn main(){
    let args: Vec<String> = env::args().collect();

    let first_size = args[1].parse::<usize>().unwrap();
    let mut last_size = first_size;
    if args.len() == 3{
        last_size = args[2].parse::<usize>().unwrap();
    }

    for i in first_size..last_size+1 {
        run_full_gray(&i);
    }
}