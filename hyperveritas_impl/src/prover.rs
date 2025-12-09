#![allow(warnings)]

use ark_ec::pairing::Pairing;
use core::num;
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
use super::helper::*;
use super::image::*;
use ark_ff::One;
use ark_std::{rand::RngCore as R, test_rng};
use itertools::Itertools;


pub const irredPolyTable: &[u32] = &[
    0, 0, 7, 11, 19, 37, 67, 131, 285, 529, 1033, 2053, 4179, 8219, 16707, 32771, 69643, 131081,
    262273, 524327, 1048585, 2097157, 4194307, 8388641, 16777351, 33554441, 67108935,
];

pub fn galoisifyPt<F: PrimeField>(nv: u32, galoisRep: u32, pt: Vec<F>)->(Vec<F>,Vec<F>,F){
    let mut myIndexes = Vec::new();
    let mut galoisRepTemp = galoisRep;

    for i in 0..nv+1{
        if galoisRepTemp >=  (2 as u32).pow((nv-(i) ).try_into().unwrap()){
            galoisRepTemp -= (2 as u32).pow((nv-(i)).try_into().unwrap()) ;
            myIndexes.push(nv-i); 
        }
    }
    let mut fiddledPt = Vec::new();
    let mut startIndex= F::zero();
    let mut zeroPt = Vec::new();
    
    for i in 1..nv{
        zeroPt.push( pt[i as usize]);
        if myIndexes.contains(&i){
            fiddledPt.push(F::one()-pt[i as usize]);
        }
        else{
            fiddledPt.push(pt[i as usize]);
        }
    }
    zeroPt.push(F::zero());
    fiddledPt.push(F::one());
    startIndex = pt[0];

    return( fiddledPt,zeroPt, startIndex) 
}
pub fn multsetProverIOP<F: PrimeField, E, PCS>(
    nv: usize,
    p1: &[Arc<DenseMultilinearExtension<F>>],
    p2: &[Arc<DenseMultilinearExtension<F>>],
    pcs_param: &PCS::ProverParam,
    ver_param: &PCS::VerifierParam,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> (
    <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::ProductCheckProof,
    Arc<DenseMultilinearExtension<F>>,
    Arc<DenseMultilinearExtension<F>>,
    VPAuxInfo<F>,

)
where
    E: Pairing<ScalarField = F>,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<F>,
        Evaluation = F,
    >,
{
    //We make the final query point for prod check.
    //fx and gx store the polynomials f(x)+r and g(x)+r
    let alpha = transcript.get_and_append_challenge(b"alpha").unwrap();
    
    let constVec = vec![alpha; 1 << nv];
    let constFuncAsPoly = Arc::new(DenseMultilinearExtension::from_evaluations_slice(
        nv, &constVec,
    ));
    let mut fx: Vec<Arc<DenseMultilinearExtension<F>>> = vec![Arc::new(p1[0].as_ref() + constFuncAsPoly.as_ref())];
    let mut gx = vec![Arc::new(p2[0].as_ref() + constFuncAsPoly.as_ref())];

    //  P = r0 + p1 + r1 * p2 
    //  Q = r0 + q1 + r1 * q2
    // Want to prove that PI_B P(B) = PI_B Q(B)
    //For multiple polynomials, we want fx to be r_0f_0x(x) + r_1f_1(x) +...+ r_nf_n(x), likewise gx
    //We add if we have multiple polynomials...
    for i in 1..(p1.len()) {
        //We get new random challenges each time
        let alpha = transcript.get_and_append_challenge(b"alpha").unwrap();
        
        let mut p1_plus_r = Vec::new();
        let mut p2_plus_r = Vec::new();
        let p1iEvals = p1[i].to_evaluations();
        let p2iEvals = p2[i].to_evaluations();

        //We now generate r_i * f_i
        for j in 0..p1[0].to_evaluations().len() {
            p1_plus_r.push(p1iEvals[j] * alpha);
            p2_plus_r.push(p2iEvals[j] * alpha);
        }
        let p1_j_plus_r_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            nv, p1_plus_r,
        ));
        let p2_j_plus_r_poly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            nv, p2_plus_r,
        ));

        // fx contains one poly, we have it as list for productcheck. This is simply fx += (r_i)(f_ix)
        fx[0] = Arc::new(fx[0].as_ref() + p1_j_plus_r_poly.as_ref());
        gx[0] = Arc::new(gx[0].as_ref() + p2_j_plus_r_poly.as_ref());
    }
    //We now prove the productcheck. We take a copy of the transcript at this point in time.
    let (proof, prod_x, frac_poly) =
        <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::prove(&pcs_param, &fx, &gx, transcript)
            .unwrap();

    //We return the prodcheck proof, as well as the prod and frac polynomials.
    
    let mut myProd1 = F::one();
    for i in 0..fx[0].evaluations.len() {
        myProd1 *= fx[0].evaluations[i];
    }
    let mut myProd2 = F::one();
    for i in 0..gx[0].evaluations.len() {
        myProd2 *= gx[0].evaluations[i];
    }

    let mut polyProd = VirtualPolynomial::new_from_mle(&fx[0], F::one());
    polyProd.mul_by_mle(gx[0].clone(), F::one());
    let polyProdInfo = polyProd.aux_info;

    return (proof, prod_x, frac_poly, polyProdInfo);
}

pub fn range_checkProverIOP<F: PrimeField, E, PCS>(
    nv: usize,
    maxVal: u32,
    p1: Arc<DenseMultilinearExtension<F>>,
    primPolyForT: u64, 
    primPolyForH: u64,
    transcript: &mut IOPTranscript<E::ScalarField>,
    pcs_param: &PCS::ProverParam,
    ver_param: &PCS::VerifierParam,
) -> (
    <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::ProductCheckProof,
    Arc<DenseMultilinearExtension<F>>,
    Arc<DenseMultilinearExtension<F>>,
    Arc<DenseMultilinearExtension<F>>,
    VPAuxInfo<F>
)
where
    E: Pairing<ScalarField = F>,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<F>,
        Evaluation = F,
    >,
{
    // We make the table and coressponding +1 table
    let mut embeddedTable: Vec<F> = vec![F::zero(); 1 << nv];
    let mut plusOneTable: Vec<F> = vec![F::zero(); 1 << nv];

    // This takes the coefficients of our poly that aren't the most significant one.
    let galoisRep = (primPolyForT) - (1 << nv);
    let galoisRepH = (primPolyForH) - (1 <<( nv+1));

    let size = 1 << nv;
    let mut binaryString: u64 = 1;

    //We create the table by setting index i to g^i(1) where g is our generator.
    for i in 1..(maxVal as usize + 1) {
        //We set T_{g^i(1)}=T_i=i
        embeddedTable[binaryString as usize] =
            F::from_le_bytes_mod_order(&transform_u32_to_array_of_u8(i as u32));
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
        plusOneTable[binaryString as usize] =
            F::from_le_bytes_mod_order(&transform_u32_to_array_of_u8(i as u32));
    }

    //We make the big h and corresponding +1 vector
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------------

    //
    let mut hTable: Vec<usize> = vec![0; (maxVal + 1).try_into().unwrap()];
    //We recall in hyperplonk that for h, they need to count how many times each element of the vector(in our case, image) appears in the table. This code creates a table that encapsulates this.

    for i in 0..p1.evaluations.len() {
        let mut b = p1.evaluations[i].to_string();
        if (b == "") {
            // println!("YES!");
            b = "0".to_string();
        } 

        hTable[b.parse::<usize>().unwrap()] += 1;
    }

    let mut embeddedH: Vec<F> = vec![F::zero(); 1 << (nv+1)];
    let mut plusOneEmbeddedH: Vec<F> = vec![F::zero(); 1 << (nv+1)];
    let size = 1 << (nv+1);
    let mut binaryString: u64 = 1;

    //We create the table by setting index i to g^i(1) where g is our generator.
    let mut counter = 0;
    for a in &hTable {
        for i in 0..(*a + 1) {
            embeddedH[binaryString as usize] =
                F::from_le_bytes_mod_order(&transform_u32_to_array_of_u8(counter));
            binaryString <<= 1;

            //If we have overflow
            if (binaryString & size != 0) {
                binaryString ^= galoisRepH;
            }
            //We remove overflow
            binaryString = (size - 1) & binaryString;
            //Binarystring is now g^i(1).
            //We set table_{g^i(1)}= T_i.
            plusOneEmbeddedH[binaryString as usize] =
                F::from_le_bytes_mod_order(&transform_u32_to_array_of_u8(counter));
        }

        if (counter < maxVal) {
            counter += 1;
        }
    }
    //--------------------------------------------------------------------EMBEDDINGS ARE DONE----------------------------------------------------------------------------
    //We now make the appropriate polynomials

    let definingH: DenseMultilinearExtension<F> = DenseMultilinearExtension::from_evaluations_vec(nv+1, embeddedH);
    let definingPlusOneH: DenseMultilinearExtension<F> = DenseMultilinearExtension::from_evaluations_vec(nv+1, plusOneEmbeddedH);
    
    let polyEmbeddedH =Arc::new(definingH.clone());
    let polyPlusOneEmbeddedH = Arc::new(definingPlusOneH);

    let polyTable = DenseMultilinearExtension::from_evaluations_vec(nv, embeddedTable);

    let polyPlusOneTable =
        DenseMultilinearExtension::from_evaluations_vec(nv, plusOneTable);

    let g1 = merge_polynomials::<F>(&[p1.clone(), Arc::new(polyTable)]).unwrap();
    let g2 = merge_polynomials::<F>(&[p1, Arc::new(polyPlusOneTable)]).unwrap();

    let (multsetProof, fx, gx, poly_infoProd) = multsetProverIOP::<F, E, PCS>(
        nv + 1,
        &[g1, g2],
        &[polyEmbeddedH, polyPlusOneEmbeddedH],
        &pcs_param,
        &ver_param,
        transcript,
    );

    return (multsetProof, fx, gx, Arc::new(definingH),poly_infoProd);
}

pub fn cropProveAffineIOP<F: PrimeField>(
    nvOrig: usize,
    nvCrop: usize,
    origImgR: Arc<DenseMultilinearExtension<F>>,
    origImgG: Arc<DenseMultilinearExtension<F>>,
    origImgB: Arc<DenseMultilinearExtension<F>>,
    croppedImgR: Arc<DenseMultilinearExtension<F>>,
    croppedImgG: Arc<DenseMultilinearExtension<F>>,
    croppedImgB: Arc<DenseMultilinearExtension<F>>,
    origWidth: usize,
    origHeight: usize,
    startX: usize,
    startY: usize,
    endX: usize,
    endY: usize,
    transcript: &mut IOPTranscript<F>,
) -> (
    [<PolyIOP<F> as SumCheck<F>>::SumCheckProof;3],
    VPAuxInfo<F>
    )
{
    let mut rng = test_rng();
    let width = endX - startX;
    let height = endY - startY;
    //create permutation
    let mut cropPerm = Vec::new();

    for i in 0..1 << nvOrig {
        let mut row = Vec::new();
        cropPerm.push(row);
    }
    let mut counter = 0;
    let mut initVal = origWidth * (startY) + (startX);

    for i in 0..(height) {
        for j in 0..(width) {
            cropPerm[initVal].push((counter, F::one()));
            counter += 1;
            initVal += 1;
        }
        initVal += origWidth - (width );
    }

    let frievaldRandVec = transcript.get_and_append_challenge_vectors(b"frievald", 1<<nvCrop).unwrap();


    let now = Instant::now();

    let permTimesR = matSparseMultVec::<F>(1 << nvOrig, 1 << nvCrop, &cropPerm, &frievaldRandVec);
    
    let elapsed_timeProver = now.elapsed();
    println!(
        "Verifier/Prover time in PermTimesR is {} seconds.",
        elapsed_timeProver.as_millis() as f64 / 1000 as f64
    );

    let now = Instant::now();

    let permTimesRPoly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        nvOrig,
        permTimesR,
    ));
    let elapsed_timeProver = now.elapsed();
    println!(
        "Time to turn perm times R into poly {} seconds.",
        elapsed_timeProver.as_millis() as f64 / 1000 as f64
    );

    let mut IPermR = VirtualPolynomial::new_from_mle(&permTimesRPoly, F::one());
    let mut IPermG = VirtualPolynomial::new_from_mle(&permTimesRPoly, F::one());
    let mut IPermB = VirtualPolynomial::new_from_mle(&permTimesRPoly, F::one());

    //The poly matrixCombined is F_M(r,b)*Im(b) (NO MORE CHANGES)
    IPermR.mul_by_mle(origImgR, F::one());
    IPermG.mul_by_mle(origImgG, F::one());
    IPermB.mul_by_mle(origImgB, F::one());

    let now = Instant::now();

    let proof0 = <PolyIOP<F> as SumCheck<F>>::prove(&IPermR, transcript).unwrap();
    let proof1 = <PolyIOP<F> as SumCheck<F>>::prove(&IPermG, transcript).unwrap();
    let proof2 = <PolyIOP<F> as SumCheck<F>>::prove(&IPermB, transcript).unwrap();

    let elapsed_timeProver = now.elapsed();
    println!(
        "Time to run sumcheck IOP is {} seconds.",
        elapsed_timeProver.as_millis() as f64 / 1000 as f64
    );
    return ([proof0, proof1, proof2], IPermR.aux_info);
}

pub fn cropProveOneAffineIOP<F: PrimeField>(
    nvOrig: usize,
    nvCrop: usize,
    origImgR: Arc<DenseMultilinearExtension<F>>,
    croppedImgR: Arc<DenseMultilinearExtension<F>>,
    origWidth: usize,
    origHeight: usize,
    startX: usize,
    startY: usize,
    endX: usize,
    endY: usize,
    transcript: &mut IOPTranscript<F>,
) -> (
    <PolyIOP<F> as SumCheck<F>>::SumCheckProof
    )
{
    let mut rng = test_rng();
    let width = endX - startX;
    let height = endY - startY;
    //create permutation
    let mut cropPerm = Vec::new();

    for i in 0..1 << nvOrig {
        let mut row = Vec::new();

        cropPerm.push(row);
    }
    let mut counter = 0;
    let mut initVal = origWidth * (startY) + (startX);
    for i in 0..(height) {
        for j in 0..(width) {
            cropPerm[initVal].push((counter, F::one()));
            counter += 1;
            initVal += 1;
        }
        initVal += origWidth - (width + 1);
    }
    
    let frievaldRandVec =transcript.get_and_append_challenge_vectors(b"frievald", 1<<nvCrop).unwrap();

    let now = Instant::now();
    let permTimesR = matSparseMultVec::<F>(1 << nvOrig, 1 << nvCrop, &cropPerm, &frievaldRandVec);
    let elapsed_timeProver = now.elapsed();
    println!(
        "Verifier/Prover time in PermTimesR is {} seconds.",
        elapsed_timeProver.as_millis() as f64 / 1000 as f64
    );

    let now = Instant::now();

    let permTimesRPoly = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
        nvOrig,
        permTimesR,
    ));
    let elapsed_timeProver = now.elapsed();
    println!(
        "Time to turn perm times R into poly {} seconds.",
        elapsed_timeProver.as_millis() as f64 / 1000 as f64
    );

    let mut IPermR = VirtualPolynomial::new_from_mle(&permTimesRPoly, F::one());

    //The poly matrixCombined is F_M(r,b)*Im(b) (NO MORE CHANGES)
    IPermR.mul_by_mle(origImgR.clone(), F::one());

   
    let now = Instant::now();
    let proof0 = <PolyIOP<F> as SumCheck<F>>::prove(&IPermR, transcript).unwrap();

    let elapsed_timeProver = now.elapsed();
    println!(
        "Time to run sumcheck IOP is {} seconds.",
        elapsed_timeProver.as_millis() as f64 / 1000 as f64
    );
    return proof0;
}

pub fn main(){
    
}