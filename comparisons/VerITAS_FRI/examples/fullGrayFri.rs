#![allow(warnings)]

use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::field::types::Field;
use plonky2::field::extension::Extendable;
use plonky2::util::timing::TimingTree;
use plonky2::field::fft::fft_root_table;
use plonky2::util::{log2_ceil, log2_strict};
use core::cmp::max;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::thread;
use rayon::prelude::*;
use plonky2::plonk::config::GenericHashOut;
use sha256::digest;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::fri::structure::FriPolynomialInfo;
use plonky2::fri::structure::FriInstanceInfo;
use plonky2::fri::structure::FriBatchInfo;
use plonky2::iop::challenger::Challenger;
use plonky2::fri::structure::FriOracleInfo;
use plonky2::fri::reduction_strategies::FriReductionStrategy;
use plonky2::fri::FriConfig;
use plonky2::fri::structure::FriOpeningBatch;
use plonky2::fri::structure::FriOpenings;
use plonky2::fri::verifier::verify_fri_proof;
use plonky2::field::extension::quadratic::QuadraticExtension;
use plonky2::field::types::PrimeField64;
use plonky2::field::types::Field64;
use plonky2::fri::proof::FriProof;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::hash::merkle_tree::MerkleCap;
use plonky2::field::fft::FftRootTable;
use std::env;
use std::io;
use std::fs::{read, OpenOptions};
use std::io::{Write, Result};

static name : &str = "fullSystemGray_Hash";

static EXPONENT : u32 = 8;
static PIXEL_RANGE : i32 = 2_i32.pow(EXPONENT);
static HASH_LENGTH : usize = 128;
const X : u128 = 3091352403337663489;
const A_CONSTANT : u128 = 3935559000370003845;
const C_CONSTANT : u128 = 2691343689449507681;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
const USE_ZK : bool = false;
mod threeGray;
use threeGray::*;
mod oneHash;
use oneHash::*;

fn write_bytes_to_file(bytes: Vec<u8>, path: &str) {
    let mut file = File::create(path).unwrap();
    file.write_all(&bytes).unwrap();
}

fn read_bytes_from_file(path: &str) -> Vec<u8> {
    let bytes = read(path).unwrap();
    bytes
}

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
fn get_sha256_of_commitments(merkle_cap: &MerkleCap<F, PoseidonHash>, instance_hash: &str, num_elements: usize) -> Vec<F> {
    let mut byte_vec = Vec::new();

    for hash in &merkle_cap.0 {
        let h = hash.to_vec();
        for elem in h {
            byte_vec.append(&mut elem.to_canonical_u64().to_le_bytes().to_vec());
        }
    }

    let s = format!("{:?}{:?}", &byte_vec, instance_hash);
    let mut val = digest(s);

    let mut ret = Vec::new();

    for _ in 0..num_elements/4 {
        let sha2561 = u64::from_str_radix(&val[0..16], 16).unwrap() % F::ORDER;
        ret.push(F::from_canonical_u64(sha2561));
        let sha2562 = u64::from_str_radix(&val[16..32], 16).unwrap();
        ret.push(F::from_canonical_u64(sha2562));
        let sha2563 = u64::from_str_radix(&val[32..48], 16).unwrap();
        ret.push(F::from_canonical_u64(sha2563));
        let sha2564 = u64::from_str_radix(&val[48..64], 16).unwrap();
        ret.push(F::from_canonical_u64(sha2564));
        val = digest(val);
    }
    
    return ret;
}

fn get_fri_config(rate_bits: usize, cap_height: usize) -> FriConfig {
    return FriConfig {
        rate_bits: rate_bits,
        cap_height: cap_height,
        proof_of_work_bits: 16,
        reduction_strategy: FriReductionStrategy::ConstantArityBits(4, 5),
        num_query_rounds: 28,
    };
}

fn prove(start: u128, old_last: u128, rate_bits: usize, cap_height: usize, omega: F, fft_root_table: &FftRootTable<F>, postfix: &str, size: &usize, pixels: usize, degree: usize) 
-> (FriOpenings<F, D>, [MerkleCap<GoldilocksField, PoseidonHash>; 3], FriProof<F, <PoseidonGoldilocksConfig as GenericConfig<D>>::Hasher, D>, u128)  {
    let mut last = old_last;
    let mut x = X;


    // w_vals = [0, 1,...,PIXEL_RANGE - 1]
    let mut w_vals = Vec::new();
    for i in 0..PIXEL_RANGE {
        let i_in_fr = GoldilocksField(i as u64);
        w_vals.push(i_in_fr);
    }

    let mut w_vals = Vec::new();
    for i in 0..PIXEL_RANGE {
        let i_in_fr = GoldilocksField(i as u64);
        w_vals.push(i_in_fr);
    }

    for _ in 0..degree - (PIXEL_RANGE as usize) {
        w_vals.push(F::ZERO);
    }

    // w[X] = poly(w_vals)
    let w = PolynomialValues::new(w_vals).ifft();

    last = print_time_since(start, last, "w interpolation done", size, postfix); 

    // v_vals = [pixel_0,...,pixel_{D-1}]
    let mut v_vals = Vec::new();
    // z_vals = [sort(v || w)]
    let mut z_vals = Vec::new();

    // reading in photo pixels...
    let file = read_photo("./images/Veri", postfix, size);
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<i32>().unwrap();

        let v_point = GoldilocksField(i as u64); 
        v_vals.push(v_point);
        z_vals.push(i);
    }

    for _ in 0..degree - pixels {
        v_vals.push(F::ZERO);
    }

    for i in 0..PIXEL_RANGE {
        z_vals.push(i);
    }

    // pad z_vals so that [z[omega*x] - z[x][1 - (z[omega*x] - z[x])] = 0 still holds true
    let z_vals_length = z_vals.len();
    for _ in 0..degree - z_vals_length {
        z_vals.push(PIXEL_RANGE - 1);
    }
    z_vals.sort();

    let mut z_f_vals = Vec::new();
    for i in 0..z_vals.len() {
        z_f_vals.push(GoldilocksField(z_vals[i] as u64));
    }
    

    // v[X] = poly(v_vals)
    let v = PolynomialValues::new(v_vals).ifft();
    // z[X] = poly(z_vals)
    let z = PolynomialValues::new(z_f_vals.clone()).ifft();
    last = print_time_since(start, last, "z and v interpolation done", size, postfix); 

    

    let mut values_vec_0 = Vec::new();
    values_vec_0.push(w.clone());
    values_vec_0.push(v.clone());
    values_vec_0.push(z.clone());

    last = print_time_since(start, last, "polynomial push done", size, postfix); 

    let commit0 = PolynomialBatch::<F, C, D>::from_coeffs(
            values_vec_0,
            rate_bits,
            USE_ZK,
            cap_height,
            &mut TimingTree::default(),
            Some(&fft_root_table),
        );
        
    last = print_time_since(start, last, "commit0 done", size, postfix); 
    let gamma = get_sha256_of_commitments(&commit0.merkle_tree.cap, "", 4)[0];

    last = print_time_since(start, last, "gamma done", size, postfix); 

    // Permutation argument
    // We want to prove:
    //           product_{i=0}^{D-1}(v_i + gamma) * product_{i=0}^{PIXEL_RANGE-1}(w_i + gamma) = product_{i=0}^{D + PIXEL_RANGE - 1}(z_i + gamma) 
    // where v holds the image pixels, w is the range that the pixel values must lie in [0, PIXEL_RANGE-1],
    // and z is the sorted concatentation of v and w

    let mut values_vec_1 = Vec::new();

    // w_prod_vals = [1, (gamma), [(gamma)(1 + gamma)],...,[(gamma)...(PIXEL_RANGE - 1 + gamma)]]
    let mut w_prod_vals = Vec::new();
    let mut product = F::ONE;
    w_prod_vals.push(product);

    for i in 0..PIXEL_RANGE {
        let i_in_fr = GoldilocksField(i as u64);
        product *= i_in_fr + gamma;
        w_prod_vals.push(product);
    }

    let w_prod_vals_len = w_prod_vals.len();
    for _ in 0..degree - w_prod_vals_len {
        product *= gamma;
        w_prod_vals.push(product);
    }
    
    // w_prod_omega_vals = [(gamma), [(gamma)(1 + gamma)],...,[(gamma)...(PIXEL_RANGE + gamma)], 1]
    let mut w_prod_omega_vals = Vec::new();
    for i in 1..w_prod_vals.len() {
        w_prod_omega_vals.push(w_prod_vals[i]);
    }
    w_prod_omega_vals.push(w_prod_vals[0]);

    let w_prod = PolynomialValues::new(w_prod_vals).ifft();

    let w_prod_omega = PolynomialValues::new(w_prod_omega_vals).ifft();

    last = print_time_since(start, last, "w_prod and w_prod_omega interpolation done", size, postfix); 

    let mut n_1_coeffs = Vec::new();
    n_1_coeffs.push(omega.exp_u64((degree - 1) as u64));
    n_1_coeffs.push(F::ZERO - F::ONE);
    
    let n_1 = PolynomialCoeffs::from(n_1_coeffs);
    // println!("n_1 eval {:?}", n_1.eval(omega.exp_u64((DEGREE - 1) as u64)));
    last = print_time_since(start, last, "n_1 interpolation done", size, postfix); 

    let mut gamma_coeffs = Vec::new();
    gamma_coeffs.push(gamma);
    let gamma_poly = PolynomialCoeffs::from(gamma_coeffs);

    // let (q_w, r_w) = (&(&w_prod_omega - &(&w_prod * &(&gamma_poly + &w))) * &n_1).div_rem(&vanishing_poly);
    let sum = &(&w_prod_omega - &(&w_prod * &(&gamma_poly + &w))) * &n_1;
    let q_w = PolynomialCoeffs::new(sum.coeffs[0..degree].to_vec());
    last = print_time_since(start, last, "q_w division done", size, postfix); 

    // Will commit to w_prod[X], q_w[X]
    values_vec_1.push(w_prod);
    values_vec_1.push(q_w);

    // v_prod_vals = [1, (pixel_0 + gamma), [(pixel_0 + gamma)(pixel_1 + gamma)],...,[(pixel_0 + gamma)...(pixel_{D-1} + gamma)]]
    let mut v_prod_vals = Vec::new();
    let mut product = F::ONE;
    v_prod_vals.push(product);

    // reading in photo pixels...
    let file = read_photo("./images/Veri", postfix, size);
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<i32>().unwrap();

        let v_point = GoldilocksField(i as u64); 

        product *= v_point + gamma;
        v_prod_vals.push(product);
    }

    for _ in 0..degree - pixels - 1 {
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
    let v_prod = PolynomialValues::from(v_prod_vals).ifft();

    // v_prod_omega[X] = v_prod[omega*X]
    let v_prod_omega = PolynomialValues::from(v_prod_omega_vals).ifft(); 

    last = print_time_since(start, last, "v_prod and v_prod_omega interpolation done", size, postfix); 

    // q_v[X] = (v_prod[omega * X] - (v_prod[X] * (gamma + v[X]))) * n_1[X] / Z_H[X]
    // let (q_v, r_v) = (&(&v_prod_omega - &(&v_prod * &(&gamma_poly + &v))) * &n_1).div_rem(&vanishing_poly);
    let sum = &(&v_prod_omega - &(&v_prod * &(&gamma_poly + &v))) * &n_1;
    let q_v = PolynomialCoeffs::new(sum.coeffs[0..degree].to_vec());

    last = print_time_since(start, last, "q_v division done", size, postfix); 

    // Will commit to v_prod[X], q_v[X]
    values_vec_1.push(v_prod);
    values_vec_1.push(q_v);

    // z_prod_vals = [1, z_vals_0 + gamma, [(z_0 + gamma)(z_vals_1 + gamma)],...,[(z_vals_0 + gamma)...(z_vals_{PIXEL_RANGE + D - 1} + gamma)]]
    let mut z_prod_vals = Vec::new();
    let mut product = F::ONE;
    z_prod_vals.push(product);
    for i in 0..z_f_vals.len() - 1 {
        product *= z_f_vals[i] + gamma;
        z_prod_vals.push(product);
    }

    // Range argument
    // We want to prove for the z constructed above that:
    //      (z[X] - z[omega*X])(1 - (z[X] - z[omega*X]) = 0 mod Z_H[X]

    // z_omega_vals = [z_vals_0 + gamma,...,[(z_vals_0 + gamma)...(z_vals_{PIXEL_RANGE + D - 1} + gamma)], 1]
    let mut z_omega_vals = Vec::new();
    for i in 1..z_vals.len() {
        z_omega_vals.push(z_f_vals[i]);
    }
    z_omega_vals.push(z_f_vals[0]);

    // z_prod_omega_vals = [z_vals_0 + gamma, [(z_vals_0 + gamma)(z_vals_1 + gamma)],...,[(z_vals_0 + gamma)...(z_vals_{PIXEL_RANGE + D - 1} + gamma)], 1]
    let mut z_prod_omega_vals = Vec::new();
    for i in 1..z_prod_vals.len() {
        z_prod_omega_vals.push(z_prod_vals[i]);
    }
    z_prod_omega_vals.push(z_prod_vals[0]);

    // for all i \in [1, PIXEL_RANGE + D], z_prod[omega^i] = \prod_{j=0}^{i-1}(z_j + gamma)
    let z_prod = PolynomialValues::from(z_prod_vals).ifft();

    // z_prod_omega[X] = z_prod[omega*X]
    let z_prod_omega = PolynomialValues::from(z_prod_omega_vals).ifft();
    last = print_time_since(start, last, "z_prod and z_prod_omega interpolation done", size, postfix); 

    // q_z[X] = (z_prod[omega * X] - (z_prod[X] * (gamma + z[X]))) * n_1[X] / Z_H[X]
    // let (q_z, r_z) = (&(&z_prod_omega - &(&z_prod * &(&gamma_poly + &z))) * &n_1).div_rem(&vanishing_poly);
    let sum = &(&z_prod_omega - &(&z_prod * &(&gamma_poly + &z))) * &n_1;
    let q_z = PolynomialCoeffs::new(sum.coeffs[0..degree].to_vec());
    last = print_time_since(start, last, "q_z division done", size, postfix); 

    let z_omega = PolynomialValues::from(z_omega_vals).ifft();

    let mut one_coeffs = Vec::new();
    one_coeffs.push(F::ONE);
    
    let one = PolynomialCoeffs::from(one_coeffs);
    last = print_time_since(start, last, "one interpolation done", size, postfix); 

    let sum = &(&(&z_omega - &z) * &(&one - &(&z_omega - &z))) * &n_1;
    let q_range = PolynomialCoeffs::new(sum.coeffs[0..degree].to_vec());

    last = print_time_since(start, last, "q_range division done", size, postfix); 

    // Will commit to z_prod[X], q_z[X], q_range[X]
    values_vec_1.push(z_prod);
    values_vec_1.push(q_z);
    values_vec_1.push(q_range);

    let commit1 = PolynomialBatch::<F, C, D>::from_coeffs(
            values_vec_1,
            rate_bits,
            USE_ZK,
            cap_height,
            &mut TimingTree::default(),
            Some(&fft_root_table),
    );

    last = print_time_since(start, last, "commit1 done", size, postfix); 

    // Now we prove knowledge of actual hash value
    // Want to generate a[X] and prove that Equation 11 in Section 5.5 holds for
    // this a[X] and the v[X] generated above

    // Use commitments to generate random coefficients [r_0,...,r_{HASH_LENGTH-1}]
    // for random linear combination of sum checks
    let hash_coeffs = get_sha256_of_commitments(&commit1.merkle_tree.cap, "", HASH_LENGTH);

    // Let A be the public hashing matrix (we will generate it with a PRG)
    // a_vals = [\sum_{i=0}{HASH_LENGTH-1}r_i * A_{i, 0},...,\sum_{i=0}{HASH_LENGTH-1}r_i * A_{i, D - 1}]
    let mut a_vals = Vec::new();

    // h_sum_vals = [0, v_vals_0 * a_vals_0 ,..., \sum_{i=0}^{D - 1} v_vals_0 * a_vals_0]
    let mut h_sum_vals = Vec::new();

    // h_sum_omega_vals = [\sum_{i=0}^{1} v_vals_0 * a_vals_0,...,\sum_{i=0}^{D - 1} v_vals_0 * a_vals_0, v_vals_0 * a_vals_0]
    let mut h_sum_omega_vals = Vec::new();
    h_sum_vals.push(F::ZERO);
    let mut sum = F::ZERO;

    // Re-read in pixels
    let file = read_photo("./images/Veri", postfix, size);
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<i32>().unwrap();

        let v_point = GoldilocksField(i as u64); 

        let mut a_point = F::ZERO; 
        for j in 0..hash_coeffs.len() {
            x = (A_CONSTANT * x + C_CONSTANT) & 0xffffffffffffffff; 
            let x1 = x >> 32;
            x = (A_CONSTANT * x + C_CONSTANT) & 0xffffffffffffffff; 
            let x2 = ((x & 0xffffffff00000000) + x1) & 0xffffffffffffffff;

            a_point += F::from_canonical_u64(u64::try_from(x2).unwrap() % F::ORDER) * hash_coeffs[j];
        }
        a_vals.push(a_point);

        sum += v_point * a_point;
        h_sum_vals.push(sum);
        h_sum_omega_vals.push(sum);
    }

    let a_vals_length = a_vals.len();
    for _ in 0..degree - a_vals_length {
        a_vals.push(F::ZERO);
    }

    for _ in 0..degree - pixels - 1 {
        h_sum_vals.push(sum);
        h_sum_omega_vals.push(sum);
    }
    h_sum_omega_vals.push(F::ZERO);


    // for all i \in [0, D - 1], a[omega^i] = \sum_{j=0}{HASH_LENGTH-1}r_j * A_{j, i}
    let a = PolynomialValues::from(a_vals).ifft(); 

    // for all i \in [0, D], h_sum[omega^i] = \sum_{j=0}^{i} v_vals_j * a_vals_j
    let h_sum = PolynomialValues::from(h_sum_vals).ifft(); 

    // h_sum_omega[X] = h_sum[omega*X]
    let h_sum_omega = PolynomialValues::from(h_sum_omega_vals).ifft();

    last = print_time_since(start, last, "a, h_sum, h_sum_omega interpolation done", size, postfix); 

    // q_h_sum[X] = (h_sum[omega*X] - h_sum[X] - (v[X] * a[X]))* n_1[X] / Z_H[X]
    let sum = &(&(&h_sum_omega - &h_sum) - &(&v * &a))* &n_1;
    let q_h_sum = PolynomialCoeffs::new(sum.coeffs[0..degree].to_vec());
    last = print_time_since(start, last, "q_h_sum interpolation done", size, postfix); 

    // Second set of polynomials we commit to
    let mut values_vec_2 = Vec::new();

    // Will commit to a[X], h_sum[X], q_h_sum[X]
    values_vec_2.push(a);
    values_vec_2.push(h_sum);
    values_vec_2.push(q_h_sum);

    let commit2 = PolynomialBatch::<F, C, D>::from_coeffs(
            values_vec_2,
            rate_bits,
            USE_ZK,
            cap_height,
            &mut TimingTree::default(),
            Some(&fft_root_table),
        );


    last = print_time_since(start, last, "commit2 done", size, postfix); 

    let mut challenger = Challenger::<F, <PoseidonGoldilocksConfig as GenericConfig<D>>::Hasher>::new();

    challenger.observe_cap::<<PoseidonGoldilocksConfig as GenericConfig<D>>::Hasher>(&commit0.merkle_tree.cap);
    challenger.observe_cap::<<PoseidonGoldilocksConfig as GenericConfig<D>>::Hasher>(&commit1.merkle_tree.cap);
    challenger.observe_cap::<<PoseidonGoldilocksConfig as GenericConfig<D>>::Hasher>(&commit2.merkle_tree.cap);

    let zeta = challenger.get_extension_challenge::<D>();

    let degree_bits = log2_strict(degree);

    let g = <<PoseidonGoldilocksConfig as GenericConfig<D>>::F as Extendable<D>>::Extension::primitive_root_of_unity(degree_bits);

    let commit0_polys = FriPolynomialInfo::from_range(
        0,
        0..commit0.polynomials.len(),
    );

    let commit1_polys = FriPolynomialInfo::from_range(
        1,
        0..commit1.polynomials.len(),
    );

    let commit2_polys = FriPolynomialInfo::from_range(
        2,
        0..commit2.polynomials.len(),
    );

    let all_polys = [commit0_polys, commit1_polys, commit2_polys].concat();


    let zeta_batch = FriBatchInfo {
        point: zeta,
        polynomials: all_polys.clone(),
    };

    // The Z polynomials are also opened at g * zeta.
    let zeta_next = g * zeta;
    let zeta_next_batch = FriBatchInfo {
        point: zeta_next,
        polynomials: all_polys.clone(),
    };

    let pixels_var = g.exp_u64((pixels) as u64);
    let pixels_batch = FriBatchInfo {
        point: pixels_var,
        polynomials: all_polys.clone(),
    };

    let pixel_range = g.exp_u64((PIXEL_RANGE) as u64);
    let pixel_range_batch = FriBatchInfo {
        point: pixel_range,
        polynomials: all_polys.clone(),
    };

    let pixels_plus_pixel_range = g.exp_u64((pixels + PIXEL_RANGE as usize) as u64);
    let pixels_plus_pixel_range_batch = FriBatchInfo {
        point: pixels_plus_pixel_range,
        polynomials: all_polys,
    };

    let openings = vec![zeta_batch, zeta_next_batch, pixels_batch, pixel_range_batch, pixels_plus_pixel_range_batch];

    let fri_oracles = vec![
            FriOracleInfo {
                num_polys: commit0.polynomials.len(),
                blinding: USE_ZK,
            },
            FriOracleInfo {
                num_polys: commit1.polynomials.len(),
                blinding: USE_ZK,
            },
            FriOracleInfo {
                num_polys: commit2.polynomials.len(),
                blinding: USE_ZK,
            }
        ];

    let instance = FriInstanceInfo {
        oracles: fri_oracles,
        batches: openings,
    };
    
    let fri_config = get_fri_config(rate_bits, cap_height);

    let mut challenger = Challenger::<F, <PoseidonGoldilocksConfig as GenericConfig<D>>::Hasher>::new();

    let opening_proof = PolynomialBatch::<F, C, D>::prove_openings(
        &instance,
        &[
            &commit0,
            &commit1,
            &commit2
        ],
        &mut challenger,
        &fri_config.fri_params(degree_bits, true),
        &mut TimingTree::default(),
    );

    last = print_time_since(start, last, "openings commitment done", size, postfix); 

    let merkle_caps = &[
        commit0.merkle_tree.cap.clone(),
        commit1.merkle_tree.cap.clone(),
        commit2.merkle_tree.cap.clone()
    ];

    let eval_commitment = |z: <<PoseidonGoldilocksConfig as GenericConfig<D>>::F as Extendable<D>>::Extension, c: &PolynomialBatch<F, C, D>| {
            c.polynomials
                .par_iter()
                .map(|p| p.to_extension::<D>().eval(z))
                .collect::<Vec<_>>()
    };

    let commit0_zeta_eval = eval_commitment(zeta, &commit0);
    let commit0_zeta_next_eval = eval_commitment(zeta_next, &commit0);
    let commit0_pixels_eval = eval_commitment(pixels_var, &commit0);
    let commit0_pixel_range_eval = eval_commitment(pixel_range, &commit0);
    let commit0_pixels_and_pixel_eval = eval_commitment(pixels_plus_pixel_range, &commit0);

    let commit1_zeta_eval = eval_commitment(zeta, &commit1);
    let commit1_zeta_next_eval = eval_commitment(zeta_next, &commit1);
    let commit1_pixels_eval = eval_commitment(pixels_var, &commit1);
    let commit1_pixel_range_eval = eval_commitment(pixel_range, &commit1);
    let commit1_pixels_and_pixel_eval = eval_commitment(pixels_plus_pixel_range, &commit1);

    let commit2_zeta_eval = eval_commitment(zeta, &commit2);
    let commit2_zeta_next_eval = eval_commitment(zeta_next, &commit2);
    let commit2_pixels_eval = eval_commitment(pixels_var, &commit2);
    let commit2_pixel_range_eval = eval_commitment(pixel_range, &commit2);
    let commit2_pixels_and_pixel_eval = eval_commitment(pixels_plus_pixel_range, &commit2);

    
    let zeta_batch = FriOpeningBatch {
        values: [
            commit0_zeta_eval.as_slice(),
            commit1_zeta_eval.as_slice(),
            commit2_zeta_eval.as_slice(),
        ].concat(),
    };
    
    let zeta_next_batch =  FriOpeningBatch {
        values: [
            commit0_zeta_next_eval.as_slice(),
            commit1_zeta_next_eval.as_slice(),
            commit2_zeta_next_eval.as_slice(),
        ].concat()
    };

    let pixels_batch = FriOpeningBatch {
        values: [
            commit0_pixels_eval.as_slice(),
            commit1_pixels_eval.as_slice(),
            commit2_pixels_eval.as_slice(),
        ].concat(),
    };

    let pixel_range_batch = FriOpeningBatch {
        values: [
            commit0_pixel_range_eval.as_slice(),
            commit1_pixel_range_eval.as_slice(),
            commit2_pixel_range_eval.as_slice(),
        ].concat(),
    };

    let pixels_plus_pixel_range_batch = FriOpeningBatch {
        values: [
            commit0_pixels_and_pixel_eval.as_slice(),
            commit1_pixels_and_pixel_eval.as_slice(),
            commit2_pixels_and_pixel_eval.as_slice(),
        ].concat(),
    };

    let fri_openings = FriOpenings {
        batches: vec![zeta_batch, zeta_next_batch, pixels_batch, pixel_range_batch, pixels_plus_pixel_range_batch],
    };

    last = print_time_since(start, last, "eval commitments done", size, postfix); 

    return (fri_openings, merkle_caps.clone(), opening_proof, last);
}

fn get_hash_proof_size(fri_open: &FriOpenings<F, D>, cap: &[MerkleCap<GoldilocksField, PoseidonHash>; 3], fri_proof: &FriProof<F, <PoseidonGoldilocksConfig as GenericConfig<D>>::Hasher, D>) -> (usize, usize) {
    
    let mut total_quadratic_extension_elems = 0;
    let mut total_goldilocks_elems = 0;
    

    // get number of field elements from fri_open
    let batches = &fri_open.batches;
    for batch in batches {
        let vals = &batch.values;
        total_quadratic_extension_elems += vals.len();
    }

    // get number of field elements from cap
    for i in 0..3 {
        for j in 0..cap[i].0.len() {
            total_goldilocks_elems += cap[i].0[j].elements.len();
        }
    }

    // now get number of field elements from fri_proof

    // add total number of extension elems from final_poly
    total_quadratic_extension_elems += fri_proof.final_poly.coeffs.len();

    // add one from fri_proof.pow_witness
    total_goldilocks_elems += 1; 

    let query_pfs = &fri_proof.query_round_proofs;

    for query_pf in query_pfs {
        let tree_pf = &query_pf.initial_trees_proof.evals_proofs;

        for eval_pf in tree_pf{
            // add goldilocks elements from eval_pf
            total_goldilocks_elems += eval_pf.0.len();
            
            // add merkle proof elements
            let merkle_proof = &eval_pf.1;
            for hash_elem in &merkle_proof.siblings {
                total_goldilocks_elems += hash_elem.elements.len();
            }
        }

        let steps = &query_pf.steps;

        for step in steps{
            
            // add quad extension elements from the evals here
            let evals_step = &step.evals;
            total_quadratic_extension_elems += evals_step.len();

            // add merkle proof elements
            let merkle_proof_step = &step.merkle_proof;
            for hash_elem in &merkle_proof_step.siblings {
                total_goldilocks_elems += hash_elem.elements.len();
            }
        }

    }

    return (total_goldilocks_elems, total_quadratic_extension_elems)

}

fn run_full_gray(size: &usize) {
    println!("Full System (Hash + Grayscale), VerITAS FRI. Size: 2^{:?}\n", size);
    let pixels : usize = 1 << size;
    let degree : usize = 1 << (size + 1);

    let rate_bits = 2;
    let cap_height = 4;
    let degree_bits = log2_strict(degree);
    let omega = F::primitive_root_of_unity(degree_bits);

    // START TIMING HASH PROOF
    let prover_hash_start = Instant::now();

    // red channel hash proof
    let size_1 = size+0;
    let thread_hash_proof_R = thread::spawn( move || {
        let pixels : usize = 1 << size_1;
        let degree : usize = 1 << (size_1 + 1);

        let rate_bits = 2;
        let cap_height = 4;
        let max_quotient_degree_factor = 4;
        let degree_bits = log2_strict(degree);
        let omegaR = F::primitive_root_of_unity(degree_bits);
        
        let max_fft_pointsR = 1 << (degree_bits + max(rate_bits, log2_ceil(max_quotient_degree_factor)));
        let fft_root_tableR = fft_root_table(max_fft_pointsR);
        // Prover
        println!("R Thread Running");

        let start = SystemTime::now();
        let start_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let start = start_epoch.as_millis();
        let r_proof = prove(start, start, 2, 4, omegaR, &fft_root_tableR, "R", &size_1, pixels, degree);

        println!("R Thread Done");

        r_proof
    });

    // green channel hash proof
    let size_2 = size+0;
    let thread_hash_proof_G = thread::spawn( move || {
        let pixels : usize = 1 << size_2;
        let degree : usize = 1 << (size_2 + 1);

        let rate_bits = 2;
        let cap_height = 4;
        let max_quotient_degree_factor = 4;
        let degree_bits = log2_strict(degree);
        let omegaR = F::primitive_root_of_unity(degree_bits);
        
        let max_fft_pointsR = 1 << (degree_bits + max(rate_bits, log2_ceil(max_quotient_degree_factor)));
        let fft_root_tableR = fft_root_table(max_fft_pointsR);

        // Prover
        println!("G Thread Running");

        let start = SystemTime::now();
        let start_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let start = start_epoch.as_millis();
        let g_proof = prove(start, start, 2, 4, omegaR, &fft_root_tableR, "G", &size_2, pixels, degree);

        println!("G Thread Done");

        g_proof
    });

    // blue channel hash proof
    let size_3 = size+0;
    let thread_hash_proof_B = thread::spawn( move || {
        let pixels : usize = 1 << size_3;
        let degree : usize = 1 << (size_3 + 1);

        let rate_bits = 2;
        let cap_height = 4;
        let max_quotient_degree_factor = 4;
        let degree_bits = log2_strict(degree);
        let omegaR = F::primitive_root_of_unity(degree_bits);
        
        let max_fft_pointsR = 1 << (degree_bits + max(rate_bits, log2_ceil(max_quotient_degree_factor)));
        let fft_root_tableR = fft_root_table(max_fft_pointsR);

        // Prover
        println!("B Thread Running");

        let start = SystemTime::now();
        let start_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let start = start_epoch.as_millis();
        let b_proof = prove(start, start, 2, 4, omegaR, &fft_root_tableR, "B", &size_3, pixels, degree);

        println!("B Thread Done");

        b_proof
    });

    // obtain proofs for computing proof size
    let red_hash_proof = thread_hash_proof_R.join().unwrap();
    let green_hash_proof = thread_hash_proof_G.join().unwrap();
    let blue_hash_proof = thread_hash_proof_B.join().unwrap();

    let elapsed_time_prover_hash = prover_hash_start.elapsed();
    println!("\n## Prover Runtime for RGB Hash Proofs: {:?} seconds\n", elapsed_time_prover_hash.as_millis() as f64 / 1000 as f64);

    // NOW GRAYSCALE SETUP

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
    println!("Computing Proof Sizes\n");

    // first, we get the size of the hash proofs.

    let (  red_hash_gold,   red_hash_quad) = get_hash_proof_size(  &red_hash_proof.0,   &red_hash_proof.1,   &red_hash_proof.2);
    let (green_hash_gold, green_hash_quad) = get_hash_proof_size(&green_hash_proof.0, &green_hash_proof.1, &green_hash_proof.2);
    let ( blue_hash_gold,  blue_hash_quad) = get_hash_proof_size( &blue_hash_proof.0,  &blue_hash_proof.1,  &blue_hash_proof.2);

    println!("r Hash Proof: {:?} Goldilocks, {:?} Quadratic",   red_hash_gold,   red_hash_quad);
    println!("g Hash Proof: {:?} Goldilocks, {:?} Quadratic", green_hash_gold, green_hash_quad);
    println!("b Hash Proof: {:?} Goldilocks, {:?} Quadratic",  blue_hash_gold,  blue_hash_quad);
    
    println!("");

    let total_hash_gold = red_hash_gold + green_hash_gold + blue_hash_gold;
    let total_hash_quad = red_hash_quad + green_hash_quad + blue_hash_quad;

    println!("total elems : {:?} Goldilocks, {:?} Quadratic", total_hash_gold, total_hash_quad);

    let total_hash_bytes = total_hash_gold * 8 + total_hash_quad * 16;

    println!("## total hash proof bytes : {:?} Bytes", total_hash_bytes);

    println!("\n");

    // now we get the size of the grayscale proof

    // get size of compressed proofs (this includes public inputs)
    let grayscale_proof_bytes_vec = grayscale_proof.1.to_bytes();
    let mut grayscale_proof_bytes = grayscale_proof_bytes_vec.len();

    // calculate size of public inputs
    let grayscale_public_field_elems = grayscale_proof.1.public_inputs.len();
    let grayscale_public_field_elems_bytes = grayscale_public_field_elems * 8;

    println!("## total grayscale proof size: {:?} Bytes", grayscale_proof_bytes);

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

    oneHash::verify(  red_hash_proof.0,   red_hash_proof.1,   red_hash_proof.2,   red_hash_proof.3, rate_bits, cap_height, degree_bits, omega, degree, pixels);
    oneHash::verify(green_hash_proof.0, green_hash_proof.1, green_hash_proof.2, green_hash_proof.3, rate_bits, cap_height, degree_bits, omega, degree, pixels);
    oneHash::verify( blue_hash_proof.0,  blue_hash_proof.1,  blue_hash_proof.2,  blue_hash_proof.3, rate_bits, cap_height, degree_bits, omega, degree, pixels);

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