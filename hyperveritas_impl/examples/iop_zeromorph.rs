#![allow(warnings)]

use core::num;
use proc_status::ProcStatus;
use arithmetic::bit_decompose;
use transcript::IOPTranscript;
use std::{marker::PhantomData, sync::Arc, ops::Range, array, iter};

use ark_ec::pairing::prepare_g1;
use ark_std::{rand::{RngCore as R, rngs::{OsRng, StdRng}, CryptoRng, RngCore, SeedableRng}, test_rng, };

use rand_chacha::ChaCha8Rng;

use hyperveritas_impl::types::*;

use plonkish_backend::{
    pcs::{
        Evaluation, PolynomialCommitmentScheme,
        univariate::{Fri, FriProverParams, FriVerifierParams},
        multilinear::{ZeromorphFri, ZeromorphFriProverParam, ZeromorphFriVerifierParam},
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
        goldilocksMont::GoldilocksMont, new_fields::Mersenne127, ff_255::ft127::Ft127 as F,
        expression::{CommonPolynomial, Expression, Query, Rotation}, 
        arithmetic::{BatchInvert, BooleanHypercube, Field as myField}, 
        transcript::{FieldTranscriptWrite, InMemoryTranscript, TranscriptWrite},
    },
};

type Pcs = ZeromorphFri<Fri<F,Blake2s>>;

const irredPolyTable: &[u32] = &[
    0, 0, 7, 11, 19, 37, 67, 131, 285, 529, 1033, 2053, 4179, 8219, 16707, 32771, 69643, 131081,
    262273, 524327, 1048585, 2097157, 4194307, 8388641, 16777351, 33554441, 67108935,
];

pub fn seeded_std_rng() -> impl RngCore + CryptoRng {
    StdRng::seed_from_u64(OsRng.next_u64())
}

pub fn rand_vec<F: myField>(n: usize, mut rng: impl RngCore) -> Vec<F> {
    iter::repeat_with(|| F::random(&mut rng)).take(n).collect()
}

fn batch_invert<F:myField>(v: &mut [F], coeff: &F) {
    // First pass: compute [a, ab, abc, ...]
    let mut prod = Vec::with_capacity(v.len());
    let mut tmp = F::ONE;
    for f in v.iter().filter(|f| (!f.is_zero()).into()) {
        tmp.mul_assign(f);
        prod.push(tmp);
    }

    // Invert `tmp`.
    tmp = tmp.invert().unwrap(); // Guaranteed to be nonzero.

    // Multiply product by coeff, so all inverses will be scaled by coeff
    tmp *= coeff;

    // Second pass: iterate backwards to compute inverses
    for (f, s) in v.iter_mut()
        // Backwards
        .rev()
        // Ignore normalized elements
        .filter(|f| (!f.is_zero()).into())
        // Backwards, skip last element, fill in one for last term.
        .zip(prod.into_iter().rev().skip(1).chain(Some(F::ONE)))
    {
        // tmp := tmp * f; f := tmp * s = 1/f
        let new_tmp = tmp * *f;
        *f = tmp * &s;
        tmp = new_tmp;
    }
}

fn get_index(i: usize, num_vars: usize) -> (usize, usize, bool) {
    let bit_sequence = bit_decompose(i as u64, num_vars);

    // the last bit comes first here because of LE encoding
    let x0 = project(&[[false].as_ref(), bit_sequence[..num_vars - 1].as_ref()].concat()) as usize;
    let x1 = project(&[[true].as_ref(), bit_sequence[..num_vars - 1].as_ref()].concat()) as usize;

    (x0, x1, bit_sequence[num_vars - 1])
}

fn project(input: &[bool]) -> u64 {
    let mut res = 0;
    for &e in input.iter().rev() {
        res <<= 1;
        res += e as u64;
    }
    res
}

fn create_frac_poly(fxs: &[MultilinearPolynomial<F>], gxs: &[MultilinearPolynomial<F>]) -> (MultilinearPolynomial<F>){
    let mut f_evals = vec![F::ONE; 1 << fxs[0].num_vars()];
    for fx in fxs.iter() {
        for (f_eval, fi) in f_evals.iter_mut().zip(fx.iter()) {
            *f_eval *= fi;
        }
    }
    let mut g_evals = vec![F::ONE; 1 << gxs[0].num_vars()];
    for gx in gxs.iter() {
        for (g_eval, gi) in g_evals.iter_mut().zip(gx.iter()) {
            *g_eval *= gi;
        }
    }
    batch_invert(&mut g_evals, &F::ONE);

    for (f_eval, g_eval) in f_evals.iter_mut().zip(g_evals.iter()) {
        if *g_eval == F::ZERO {
            println!("throw");
        }
        *f_eval *= g_eval;
    }

    MultilinearPolynomial::new(f_evals)
}

fn create_prod_poly<F: myField> (frac_poly: &MultilinearPolynomial<F>) -> MultilinearPolynomial<F> {
    let num_vars = frac_poly.num_vars();
    let frac_evals = &frac_poly.evals();

    // ===================================
    // prod(x)
    // ===================================
    //
    // `prod(x)` can be computed via recursing the following formula for 2^n-1
    // times
    //
    // `prod(x_1, ..., x_n) :=
    //      [(1-x1)*frac(x2, ..., xn, 0) + x1*prod(x2, ..., xn, 0)] *
    //      [(1-x1)*frac(x2, ..., xn, 1) + x1*prod(x2, ..., xn, 1)]`
    //
    // At any given step, the right hand side of the equation
    // is available via either frac_x or the current view of prod_x
    let mut prod_x_evals = vec![];
    for x in 0..(1 << num_vars) - 1 {
        // sign will decide if the evaluation should be looked up from frac_x or
        // prod_x; x_zero_index is the index for the evaluation (x_2, ..., x_n,
        // 0); x_one_index is the index for the evaluation (x_2, ..., x_n, 1);
        let (x_zero_index, x_one_index, sign) = get_index(x, num_vars);
        if !sign {
            prod_x_evals.push(frac_evals[x_zero_index] * frac_evals[x_one_index]);
        } else {
            // sanity check: if we are trying to look up from the prod_x_evals table,
            // then the target index must already exist
            if x_zero_index >= prod_x_evals.len() || x_one_index >= prod_x_evals.len() {
                println!("throw prod");
            }
            prod_x_evals.push(prod_x_evals[x_zero_index] * prod_x_evals[x_one_index]);
        }
    }

    // prod(1, 1, ..., 1) := 0
    prod_x_evals.push(F::ZERO);

    MultilinearPolynomial::new(prod_x_evals)
}


fn product_check(pp: <Pcs as PolynomialCommitmentScheme<F>>::ProverParam, fxs: MultilinearPolynomial<F>, gxs: MultilinearPolynomial<F>, 
    transcript: &mut impl TranscriptWrite<<Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>, start: usize) -> (Expression<F>, Vec<MultilinearPolynomial<F>>, Vec<F>, Vec<F>, Vec<<Fri<F, Blake2s> as PolynomialCommitmentScheme<F>>::Commitment>) 
{
    let mut rng = test_rng();

    let num_vars = fxs.num_vars();

    let frac_poly = create_frac_poly(&[fxs.clone()], &[gxs.clone()]);

    let prod_poly = create_prod_poly(&frac_poly);

    let mut p1_evals = vec![F::ZERO; 1 << num_vars];
    let mut p2_evals = vec![F::ZERO; 1 << num_vars];
    for x in 0..1 << num_vars {
        let (x0, x1, sign) = get_index(x, num_vars);
        if !sign {
            p1_evals[x] = frac_poly.evals()[x0];
            p2_evals[x] = frac_poly.evals()[x1];
        } else {
            p1_evals[x] = prod_poly.evals()[x0];
            p2_evals[x] = prod_poly.evals()[x1];
        }
    }

    let p1_poly = MultilinearPolynomial::new(p1_evals);
    let p2_poly = MultilinearPolynomial::new(p2_evals);

    let prod_exp = Expression::<F>::Polynomial(Query::new(start+1, Rotation::cur()));
    let line1 = prod_exp;

    let p1_exp = Expression::<F>::Polynomial(Query::new(start+2, Rotation::cur()));
    let p2_exp = Expression::<F>::Polynomial(Query::new(start+3, Rotation::cur()));
    let line2= Expression::<F>::Scaled(Box::new(Expression::<F>::Product(Box::new(p1_exp), Box::new(p2_exp))), -F::ONE);

    let frac_prod_coms = Pcs::batch_commit_and_write(&pp, &[frac_poly.clone(), prod_poly.clone()], transcript).unwrap();

    let beta = transcript.squeeze_challenge();
    let frac_exp = Expression::<F>::Polynomial(Query::new(start, Rotation::cur()));
    let g_exp = Expression::<F>::Polynomial(Query::new(start+5, Rotation::cur()));
    let line3 = Expression::<F>::Scaled(Box::new(Expression::<F>::Product(Box::new(frac_exp), Box::new(g_exp))), beta);

    let f_exp = Expression::<F>::Polynomial(Query::new(start+4, Rotation::cur()));
    let line4 = Expression::<F>::Scaled(Box::new(f_exp.clone()), -beta);

    let gates = line1+line2+line3+line4;
    let alpha: Expression<F> = Expression::Challenge(0);
    let eq = Expression::eq_xy(0);
    let juicer = Expression::distribute_powers(&vec![gates], &alpha) * eq;
    let polys = vec![frac_poly.clone(), prod_poly.clone(), p1_poly.clone(), p2_poly.clone(), fxs.clone(), gxs.clone()];

    let challenges = vec![transcript.squeeze_challenge()];
    let rand_vector = transcript.squeeze_challenges(num_vars);

    let ys = [rand_vector.clone()];

    (juicer.clone(), polys.clone(), challenges.clone(), rand_vector.clone(), frac_prod_coms)
}

pub fn multsetCreatePolys(
    pp: <Pcs as PolynomialCommitmentScheme<F>>::ProverParam,
    nv: usize,
    p1: &[MultilinearPolynomial<F>],
    p2: &[MultilinearPolynomial<F>],
    transcript: &mut impl TranscriptWrite<<Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>,
    start: usize
) -> (Expression<F>, Vec<MultilinearPolynomial<F>>, Vec<F>, Vec<F>, Vec<<Fri<F, Blake2s> as PolynomialCommitmentScheme<F>>::Commitment>)
{
    let alpha = transcript.squeeze_challenge();

    let mut inter_poly = Vec::new();
    let mut inter_poly2 = Vec::new();
    for i in (0..p1[0].evals().len()){
        inter_poly.push(p1[0].evals()[i] +alpha);
        inter_poly2.push(p2[0].evals()[i] + alpha);
    }

    let mut fx: Vec<MultilinearPolynomial<F>> = vec![MultilinearPolynomial::new(inter_poly)];
    let mut gx: Vec<MultilinearPolynomial<F>> = vec![MultilinearPolynomial::new(inter_poly2)];

    for i in 1..(p1.len()) {
        //We get new random challenges each time
        let alpha = transcript.squeeze_challenge();
        let mut p1_plus_r = Vec::new();
        let mut p2_plus_r = Vec::new();
        let p1iEvals = p1[i].evals();
        let p2iEvals = p2[i].evals();

        //We now generate r_i * f_i
        for j in 0..p1[0].evals().len() {
            p1_plus_r.push(p1iEvals[j] * alpha);
            p2_plus_r.push(p2iEvals[j] * alpha);
        }
        let p1_j_plus_r_poly = MultilinearPolynomial::new(p1_plus_r);
        let p2_j_plus_r_poly = MultilinearPolynomial::new(p2_plus_r);

        // fx contains one poly, we have it as list for productcheck. This is simply fx += r_if_ix
        let mut new_poly = Vec::new();
        let mut new_poly2 = Vec::new();
        for i in (0..fx[0].evals().len()){
            new_poly.push(fx[0].evals()[i] + p1_j_plus_r_poly.evals()[i]);
            new_poly2.push(gx[0].evals()[i] + p2_j_plus_r_poly.evals()[i]);
        }

        fx[0] = MultilinearPolynomial::new(new_poly);
        gx[0] = MultilinearPolynomial::new(new_poly2);
    }

    //We now prove the productcheck. We take a copy of the transcript at this point in time.

    //We return the prodcheck proof, as well as the prod and frac polynomials.
    return product_check(pp, fx[0].clone(), gx[0].clone(), transcript, start);
}

pub fn galoisifyPt(nv: u32, galoisRep: u32, pt: Vec<F>)->(Vec<F>,Vec<F>,F){
    let mut myIndexes = Vec::new();
    let mut galoisRepTemp = galoisRep;

    for i in 0..nv+1{
        if galoisRepTemp >=  (2 as u32).pow((nv-(i) ).try_into().unwrap()){
            galoisRepTemp -= (2 as u32).pow((nv-(i)).try_into().unwrap()) ;
            myIndexes.push(nv-i); 
        }
    }

    let mut fiddledPt = Vec::new();
    let mut startIndex= F::ZERO;
    let mut zeroPt = Vec::new();
    
    for i in 1..nv{
        zeroPt.push( pt[i as usize]);
        if myIndexes.contains(&i){
            fiddledPt.push(F::ONE-pt[i as usize]);
        }
        else{
            fiddledPt.push(pt[i as usize]);
        }
    }
    zeroPt.push(F::ZERO);
    fiddledPt.push(F::ONE);
    startIndex = pt[0];

    return( fiddledPt,zeroPt, startIndex) 
}


pub fn range_checkProverIOP(
    pp: <Pcs as PolynomialCommitmentScheme<F>>::ProverParam,
    nv: usize,
    maxVal: u64,
    hTable: Vec<usize>,
    p1: MultilinearPolynomial<F>,
    primPolyForT: u64,
    primPolyForH: u64,
    transcript: &mut impl TranscriptWrite<<Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>,
    start: usize
) -> (Expression<F>, Vec<MultilinearPolynomial<F>>, Vec<F>, Vec<F>, Vec<<Fri<F, Blake2s> as PolynomialCommitmentScheme<F>>::Commitment>)
{
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    let mut embeddedTable: Vec<F> = vec![F::ZERO; 1 << nv];
    let mut plusOneTable: Vec<F> = vec![F::ZERO; 1 << nv];
    //This takes the coefficients of our poly that aren't the most significant one.
    let galoisRep = (primPolyForT) - (1 << nv);
    //This is how big our table is
    let size = 1 << nv;
    let mut binaryString: u64 = 1;
    //We create the table by setting index i to g^i(1) where g is our generator.
    for i in 1..(maxVal as usize + 1) {
        //We set T_{g^i(1)}=T_i=i
        embeddedTable[binaryString as usize] = F::from(i as u64);
        //This represents a multiplication by x
        binaryString <<= 1;
        //If we have overflow, we remove it
        if (binaryString & size != 0) {
            //We utilize the equivalence relation
            binaryString ^= galoisRep;
        }
        //We remove overflow
        //Binarystring is now g^i(1).
        //We set table_{g^i(1)}= T_i. In this implementation, we assume that the maxval is less than or equal to 255.
        binaryString = (size - 1) & binaryString;
        //We now add to the plus one table.
        plusOneTable[binaryString as usize] = F::from(i as u64);
    }

    //We make the big h and corresponding +1 vector
 
    let size = 1 << (nv+1);
    let mut embeddedH: Vec<F> = vec![F::ZERO; 1 << (nv+1)];
    let mut plusOneEmbeddedH: Vec<F> = vec![F::ZERO; 1 << (nv+1)];
    let mut binaryString: u64 = 1;

    let galoisRep = (primPolyForH) - (1 << (nv+1));

    //We create the table by setting index i to g^i(1) where g is our generator.
    let mut counter = 0;
    for a in &hTable {
        for i in 0..(*a + 1) {
            //println!("binstr: {:?}", binaryString as usize);
            embeddedH[binaryString as usize] = F::from(counter);
            binaryString <<= 1;

            //If we have overflow
            if (binaryString & size != 0) {
                //We utilize the equivalence relation
                binaryString ^= galoisRep;
            }
            //We remove overflow
            binaryString = (size - 1) & binaryString;
            //Binarystring is now g^i(1).
            //We set table_{g^i(1)}= T_i.
            plusOneEmbeddedH[binaryString as usize] = F::from(counter);
        }

        if (counter < maxVal) {
            counter += 1;
        }
    }

    let polyEmbeddedH = MultilinearPolynomial::new(embeddedH);
    let polyPlusOneEmbeddedH =MultilinearPolynomial::new(plusOneEmbeddedH);

    let mut poly_emb_h_com = Pcs::batch_commit_and_write(&pp, &[polyEmbeddedH.clone()], transcript).unwrap();

    let mut g1_table = p1.evals().clone().to_vec();
    g1_table.append(&mut embeddedTable.clone());

    let mut g2_table = p1.evals().clone().to_vec();
    g2_table.append(&mut plusOneTable);

    let imPoly = MultilinearPolynomial::new(p1.evals().clone().to_vec());
    let tablePoly = MultilinearPolynomial::new(embeddedTable);

    let g1 = MultilinearPolynomial::new(g1_table);
    let g2 = MultilinearPolynomial::new(g2_table);

    let mut myResult = multsetCreatePolys(pp, nv+1, &[g1,g2], &[polyEmbeddedH.clone(), polyPlusOneEmbeddedH], transcript, start);
    myResult.4.append(&mut poly_emb_h_com);
    myResult.1.push(polyEmbeddedH);

    return myResult
}

fn main(){
   
}