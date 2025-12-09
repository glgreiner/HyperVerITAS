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
use hyperveritas_impl::helper::*;
use hyperveritas_impl::image::*;

use hyperveritas_impl::prover::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

const irredPolyTable: &[u32] = &[
    0, 0, 7, 11, 19, 37, 67, 131, 285, 529, 1033, 2053, 4179, 8219, 16707, 32771, 69643, 131081,
    262273, 524327, 1048585, 2097157, 4194307, 8388641, 16777351, 33554441, 67108935,
];

const X: u128 = 3091352403337663489;
const A_CONSTANT: u128 = 3935559000370003845;
const C_CONSTANT: u128 = 2691343689449507681;

fn hashPreimageIOP<F: PrimeField, E, PCS>(
    numCols: usize,
    numRows: usize,
    RGBEvals: [Vec<F>;3],
    transcript: &mut IOPTranscript<E::ScalarField>,
    pcs_param: &PCS::ProverParam,
    ver_param: &PCS::VerifierParam,
) -> (
    Vec<VirtualPolynomial<F>>,
    [<PolyIOP<F> as SumCheck<F>>::SumCheckProof;3],
    [VPAuxInfo<F>;3],
    Vec<<PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::ProductCheckProof>,
    Vec<Arc<DenseMultilinearExtension<F>>>,
    Vec<Arc<DenseMultilinearExtension<F>>>,
    Vec<Arc<DenseMultilinearExtension<F>>>,
    Vec<VPAuxInfo<F>>
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
    //We assume we use the randomness matrix.
    let mut rng = test_rng();
    let mut matrixA = Vec::new();
    for i in 0..128 {
        matrixA.push(ChaCha8Rng::seed_from_u64(i));
    }

    //We are given the image polynomial
    let mut imgPolies: Vec<Arc<DenseMultilinearExtension<F>>> = Vec::new();

    imgPolies.push(vec_to_poly::<F>(RGBEvals[0].clone()).0);
    imgPolies.push(vec_to_poly::<F>(RGBEvals[1].clone()).0);
    imgPolies.push(vec_to_poly::<F>(RGBEvals[2].clone()).0);

    //We make Frievald random vec
    let mut frievaldRandVec = Vec::new();

    for i in 0..(1 << numRows) {
        let alpha = transcript.get_and_append_challenge(b"alpha").unwrap();
        frievaldRandVec.push(alpha);
    }

    //We make rT*A
    let now = Instant::now();

    let mut rTA = Vec::new();

    for i in 0..(1 << numCols) {
        let mut mySum = F::zero();
        for j in 0..128 {
            mySum += F::rand(&mut matrixA[j]) * frievaldRandVec[j];
        }
        rTA.push(mySum);
    }

    let (rTAPoly, _) = vec_to_poly::<F>(rTA);
    let elapsed_time = now.elapsed();
    println!(
        "Prover time to do rTA is {:?} seconds \n",
        elapsed_time.as_millis() as f64 / 1000 as f64
    );
    //We run the sumcheck on rTA * I
    let now = Instant::now();
    let mut RHS_RGB = Vec::new();
    for i in 0..3{
        RHS_RGB.push(VirtualPolynomial::new_from_mle(&rTAPoly, F::one()));
        RHS_RGB[i].mul_by_mle(imgPolies[i].clone(), F::one());
    }

    let proofRGB = [<PolyIOP<F> as SumCheck<F>>::prove(&RHS_RGB[0], transcript).unwrap(),
        <PolyIOP<F> as SumCheck<F>>::prove(&RHS_RGB[1], transcript).unwrap(),
        <PolyIOP<F> as SumCheck<F>>::prove(&RHS_RGB[2], transcript).unwrap()];
    
    let poly_infoRGB = [RHS_RGB[0].aux_info.clone(),
        RHS_RGB[1].aux_info.clone(),
        RHS_RGB[2].aux_info.clone()];
    
    let elapsed_time = now.elapsed();
    println!(
        "Prover time to do Sumcheck for hash preimage is {:?} seconds \n",
        elapsed_time.as_millis() as f64 / 1000 as f64
    );
    let mut mySum = F::zero();

    //We run range check on image
    let now = Instant::now();
    let (mut multsetProofRGB, mut fxRGB, mut gxRGB, mut hRGB, mut poly_infoProds) = (Vec::new(),Vec::new(),Vec::new(),Vec::new(), Vec::new());
    for i in 0..3{
        let (multsetProof, fx, gx, h, poly_infoProd) = range_checkProverIOP::<F, E, PCS>(
         numCols,
         255.try_into().unwrap(),
         imgPolies[i].clone(),
         irredPolyTable[numCols].try_into().unwrap(),
         irredPolyTable[numCols+1].try_into().unwrap(),
         transcript,
         &pcs_param,
         &ver_param,
        );

        multsetProofRGB.push(multsetProof);
        fxRGB.push(fx);
        gxRGB.push(gx);
        hRGB.push(h);
        poly_infoProds.push(poly_infoProd);
    }
    let elapsed_time = now.elapsed();
    println!(
        "Prover time to do MultCheck for hash preimage is {:?} seconds \n",
        elapsed_time.as_millis() as f64 / 1000 as f64
    );

    //We return a vector containing the final points to evaluate I, the final points to evaulate h(from range check), the final points
    //to evaluate the prod and frac polynomials, as well as the sumcheck proof, range check proof.
    return (RHS_RGB, proofRGB, poly_infoRGB, multsetProofRGB, fxRGB, gxRGB, hRGB, poly_infoProds);
}

fn run_full_gray_kzg(testSize: usize) {
    let mut rng = test_rng();
    let numCols = testSize;

    let numRows = 7;
    let length = numCols + 1;

    let fileName = format!("images/Timings{}.json", testSize);
    let srs = PCS::gen_srs_for_testing(&mut rng, length).unwrap();
    let (pcs_param, ver_param) = PCS::trim(&srs, None, Some(length)).unwrap();
   
    // LOAD IMAGE----------------------------------------------------------------------------------------------------------
    let origImg = load_image(&fileName);
    println!("this is dim {:?}", origImg.cols * origImg.rows);
    //Below we do padding, prover works with padded image, but later sends the unpadded commitment to verifier (this is fine as unpadded effectively has padding as 0)
    let mut RGBEvals = [toFieldVec(&origImg.R),toFieldVec(&origImg.G),toFieldVec(&origImg.B)];

    //We implement padding
    for i in 0..(RGBEvals[0].len().next_power_of_two() - RGBEvals[0].len()) {
        RGBEvals[0].push(F::zero());
        RGBEvals[1].push(F::zero());
        RGBEvals[2].push(F::zero());

    }
    //THIS IS HASHING MATRIX CREATED BY CAMERA------------------------------------------------------------
    let mut testDigestRGB = Vec::new();
    for k in 0 ..3{
        let mut matrixA = Vec::new();
        for i in 0..128 {
            matrixA.push(ChaCha8Rng::seed_from_u64(i));
        }
        //THIS IS HASHING DONE BY CAMERA------------------------------------------------------------
        let mut testDigest = Vec::new();
        for i in 0..128 {
            let mut mySum = F::zero();
            for j in 0..(1 << numCols) {
                mySum += F::rand(&mut matrixA[i]) * RGBEvals[k][j];
            }
            testDigest.push(mySum);
        }
        testDigestRGB.push(testDigest);
    }
    //THIS IS PROVER DOING EVERYTHING
    let now0 = Instant::now();
    let origImg = load_image(&fileName);
    let mut polies = Vec::new();
    polies.push(vec_to_poly(toFieldVec(&origImg.R)).0);
    polies.push( vec_to_poly(toFieldVec(&origImg.G)).0);
    polies.push( vec_to_poly(toFieldVec(&origImg.B)).0);

    let now2 = Instant::now();
    let mut coms = Vec::new();
    for i in 0..polies.len(){
        coms.push(PCS::commit(&pcs_param,&polies[i]).unwrap());
    }

    let elapsed_time = now2.elapsed();
    println!("KZG COMMIT TIME IS {:?} seconds", elapsed_time.as_millis() as f64 / 1000 as f64);
    let nowOpens = Instant::now();



    let mut transcript =
        <PolyIOP<F> as ProductCheck<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::init_transcript(
        );
    let mut transcriptVerifier =
        <PolyIOP<F> as ProductCheck<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::init_transcript(
        );

    for i in 0..3{
        transcript.append_serializable_element(b"img(x)", &coms[i]);
        transcriptVerifier.append_serializable_element(b"img(x)", &coms[i]);
    }


    // TIME THE IOP------------------------------------------------------------------------------------
    let now: Instant = Instant::now();
    let (RHS_RGB, proofRGB, poly_info_matMult, multsetProof, prod_xRGB, frac_xRGB, hRGB, poly_infoProds) =
        hashPreimageIOP::<F, Bls12_381, MultilinearKzgPCS<Bls12_381>>(
            numCols,
            numRows,
            RGBEvals,
            &mut transcript,
            &pcs_param,
            &ver_param,
        );
    let elapsed_time = now.elapsed();
    println!(
        "Prover time to do IOP is {:?} seconds",
        elapsed_time.as_millis() as f64 / 1000 as f64
    );
        //-----------------------------------------------------------------------------------GRAYSCALE--------------------------------------------------------------------------------------------
    let now: Instant = Instant::now();

    let grayFileName = format!("images/Gray{}.json", testSize);

    let grayImgOrig = load_image(grayFileName);

    let (origImgRPoly, _) = vec_to_poly(toFieldVec::<F>(&origImg.R));
    let (origImgGPoly, _) = vec_to_poly(toFieldVec::<F>(&origImg.G));
    let (origImgBPoly, _) = vec_to_poly(toFieldVec::<F>(&origImg.B));

    let (grayImg, _)  = vec_to_poly(toFieldVec::<F>(&grayImgOrig.R));
    
    let mut grayError = Vec::new();
    for i in 0..1<<numCols{
        grayError.push(F::from_le_bytes_mod_order(&[100])*grayImg.evaluations[i]-
        (F::from_le_bytes_mod_order(&[30])*origImgRPoly.evaluations[i]+
        F::from_le_bytes_mod_order(&[59])*origImgGPoly.evaluations[i]+
        F::from_le_bytes_mod_order(&[11])*origImgBPoly.evaluations[i]
        )
        +F::from_le_bytes_mod_order(&[50])
    );
    let errorVal = 50 + 100* grayImgOrig.R[i] as i32 - (30* origImg.R[i]as i32 + 59*origImg.G[i]as i32+11*origImg.B[i]as i32);
    
    }
    
    let mut polies2 = Vec::new();
    for i in 0..3{
        polies2.push(hRGB[i].clone());
        polies2.push(prod_xRGB[i].clone());
        polies2.push(frac_xRGB[i].clone());
    }

    let mut hComs = Vec::new();
    for i in 0..3{
        let hCom = PCS::commit(&pcs_param, &hRGB[i]).unwrap();
        hComs.push(hCom);
        transcript.append_serializable_element(b"hCom(x)", &hCom);

    }

    let nowOpens = Instant::now();
    for i in 0..3{
        coms.push(hComs[i]);
        coms.push(multsetProof[i].prod_x_comm);
        coms.push(multsetProof[i].frac_comm);

    }
    // START OF GRAYSCALE STUFF
    let (grayErrPoly, _) = vec_to_poly(grayError);
    let grayErrCom = PCS::commit(&pcs_param, &grayErrPoly).unwrap();
    transcript.append_serializable_element(b"gray(x)", &grayErrCom);

    // Do range check for grayscale
    let (multsetProofGray, fxGray, gxGray, hGray, auxGray) = range_checkProverIOP::<F, Bls12_381, MultilinearKzgPCS<Bls12_381>>(
    numCols,
    100.try_into().unwrap(),
    grayErrPoly.clone(),
    irredPolyTable[numCols].try_into().unwrap(),
    irredPolyTable[numCols+1].try_into().unwrap(),
    &mut transcript,
    &pcs_param,
    &ver_param,
    );

    let elapsed_time = now.elapsed();
    println!(
        "Prover time to do GRAYSCALE IOP is {:?} seconds",
        elapsed_time.as_millis() as f64 / 1000 as f64
    );
    
    // PUSH COMS FOR GRAYSCALE ERROR
    coms.push(grayErrCom);
    let hGrayCom = PCS::commit(&pcs_param, &hGray).unwrap().clone();   

    polies2.push(grayErrPoly);
    polies2.push(hGray);
    polies2.push(gxGray);
    polies2.push(fxGray);

    coms.push(hGrayCom);
    coms.push(multsetProofGray.prod_x_comm);
    coms.push(multsetProofGray.frac_comm);

    let mut final_query = vec![F::one(); numCols+1];
    final_query[0] = F::zero();

    transcript.append_serializable_element(b"hGrayCom(x)", &hGrayCom);

    // END OF GRAYSCALE COMS COMMITS
    let mut points = Vec::new();

    // HASH PREIMAGE POINT
    for i in 0..3{
    points.push(proofRGB[i].point.clone());
    }
    // 0 vector, used for h
    points.push(vec![F::zero(); numCols+1]);
    // 1..10 vector, used for prod 
    let mut final_query = vec![F::one(); numCols+1];
    final_query[0] = F::zero();
    points.push(final_query);
    // Eval for range for image
    for i in 0.. 3{
        let myRand = &multsetProof[i].zero_check_proof.point;
        
        let mut myRandSmall = Vec::new();
        for i in 0..myRand.len()-1{
            myRandSmall.push(myRand[i]);
        }
        // myRand.push(F::rand(&mut rng));
        points.push(myRandSmall.clone());
        myRandSmall = myRand.clone();

        let galoisRep = irredPolyTable[numCols + 1] - (1 <<( numCols+1));
        let (fiddle, zero, startVal) = galoisifyPt::<F>((numCols+1) as u32, galoisRep, myRand.clone());

        // point 1 for h_{+1}
        points.push(fiddle);
        // point 2 for h_{+1}
        points.push( zero.clone());
        //Rand point used for prod and frac polies 
       
        points.push(myRand.clone());
        // Randpoint but last is 0
        let mut ptRand= Vec::new();
        ptRand.push(F::zero());
        for i in 0..myRand.clone().len()-1{
            ptRand.push(myRand[i]);
        }
        points.push(ptRand.clone());
        // Randpoint but last is 1
        ptRand[0] = F::one();
        points.push(ptRand.clone());    
    }

    // Transformation point for grayscale
    let grayTransformPt = transcript.get_and_append_challenge_vectors(b"alpha", numCols).unwrap();
    
    points.push(grayTransformPt);

    let myRand = &multsetProofGray.zero_check_proof.point;
        
    let mut myRandSmall = Vec::new();
    for i in 0..myRand.len()-1{
        myRandSmall.push(myRand[i]);
    }

    points.push(myRandSmall.clone());
    myRandSmall = myRand.clone();

    let galoisRep = irredPolyTable[numCols + 1] - (1 <<( numCols+1));
    let (fiddle, zero, startVal) = galoisifyPt::<F>((numCols+1) as u32, galoisRep, myRand.clone());
    // point 1 for h_{+1}
    points.push(fiddle);
    // point 2 for h_{+1}
    points.push( zero.clone());
    //Rand point used for prod and frac polies 
    
    points.push(myRand.clone());
    // Randpoint but last is 0
    let mut ptRand= Vec::new();
    ptRand.push(F::zero());
    for i in 0..myRand.clone().len()-1{
        ptRand.push(myRand[i]);
    }
    points.push(ptRand.clone());
    // Randpoint but last is 1
    ptRand[0] = F::one();
    points.push(ptRand.clone());    

    let mut evalPols = Vec::new();
    let mut evalPoints:Vec<Vec<F>> = Vec::new();
    let mut evalVals = Vec::new();
    let mut evalComs = Vec::new();
    let mut evalPolsBig = Vec::new();
    let mut evalPointsBig:Vec<Vec<F>>  = Vec::new();
    let mut evalValsBig = Vec::new();
    let mut evalComsBig = Vec::new();

    
    for i in 0..3{
    // //----------------------------------------------We first add opening for matrixMult for hash.----------------------------------------------    
        evalPols.push(polies[i].clone());
        evalPoints.push(points[i].clone());
        evalVals.push(polies[i].evaluate(&points[i]).unwrap());
        evalComs.push(coms[i].clone());
       
        // // //----------------------------------------------We now add alpha_range for image----------------------------------------------
        let polIndex = i;
        let ptIndex = 5+6*i;
        evalPols.push(polies[polIndex].clone());
        
        evalPoints.push(points[ptIndex].clone());
        evalVals.push(polies[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComs.push(coms[polIndex].clone());
        
        // // //WE NOW DO BIG POLYNOMIALS --------------------------------------------------------------------------------------------
         // //----------------------------------------------We now add 0 for h----------------------------------------------
        let polIndex = 3*i;
        let ptIndex = 3;
        evalPolsBig.push(polies2[polIndex].clone());
        
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(F::zero());
        evalComsBig.push(coms[polIndex+3].clone());
        // // //----------------------------------------------We now add alpha_range for h----------------------------------------------

        let polIndex = 3*i;
        let ptIndex = 8+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());
        // // //----------------------------------------------We now add alpha_range Fiddle for h----------------------------------------------

        let polIndex = 3*i;
        let ptIndex = 6+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());
        
        // // //----------------------------------------------We now add alpha_range 0 for h----------------------------------------------
        let polIndex = 3*i;
        let ptIndex = 7+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());

        // // //----------------------------------------------We then add prod for 1...10----------------------------------------------
        
        let polIndex = 1+3*i;
        let ptIndex = 4;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(F::one());
        evalComsBig.push(coms[polIndex+3].clone());

        // // //----------------------------------------------We now add alpha_range for prod----------------------------------------------
        let polIndex = 1+3*i;
        let ptIndex = 8+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());

        // // //----------------------------------------------We now add alpha_range for frac----------------------------------------------

        let polIndex = 2+3*i;
        let ptIndex = 8+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());

        // //----------------------------------------------we now add alpha_range||0 for prod----------------------------------------------
        let polIndex = 1+3*i;
        let ptIndex = 9+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());

        // // // // //----------------------------------------------we now add alpha_range||0 for frac----------------------------------------------
        let polIndex = 2+3*i;
        let ptIndex = 9+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());
        
        // // // //----------------------------------------------we now add alpha_range||1 for prod----------------------------------------------
        let polIndex = 1+3*i;
        let ptIndex = 10+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());

        // // // //----------------------------------------------we now add alpha_range||1 for frac----------------------------------------------
        let polIndex = 2+3*i;
        let ptIndex = 10+6*i;
        evalPolsBig.push(polies2[polIndex].clone());
        evalPointsBig.push(points[ptIndex].clone());
        evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComsBig.push(coms[polIndex+3].clone());   
        
        // // // //----------------------------------------------we now add transPoint for IMG ----------------------------------------------
        let polIndex = i;
        let ptIndex = 23;
        evalPols.push(polies[polIndex].clone());
        evalPoints.push(points[ptIndex].clone());
        evalVals.push(polies[polIndex].evaluate(&points[ptIndex]).unwrap());
        evalComs.push(coms[polIndex].clone());
    }

    //----------------------------------------------GRAYSCALE: We first add opening for transPoint for GRAY.----------------------------------------------    
    let polIndex = 9;
    let ptIndex = 23;
    evalPols.push(polies2[polIndex].clone());
    evalPoints.push(points[ptIndex].clone());
    evalVals.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComs.push(coms[polIndex+3].clone());
    // //----------------------------------------------GRAYSCALE: We now add 0 for h----------------------------------------------
    let polIndex = 10;
    let ptIndex = 3;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range for image----------------------------------------------
    let polIndex = 9;
    let ptIndex = 24;
    evalPols.push(polies2[polIndex].clone());
    evalPoints.push(points[ptIndex].clone());
    evalVals.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComs.push(coms[polIndex].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range for h----------------------------------------------

    let polIndex = 10;
    let ptIndex = 27;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range modified0 for h----------------------------------------------

    let polIndex = 10;
    let ptIndex = 25;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());
    // // //----------------------------------------------GRAYSCALE: We now add alpha_range modified1 for h----------------------------------------------
    let polIndex = 10;
    let ptIndex = 26;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());
    
    // // //WE NOW DO BIG POLYNOMIALS --------------------------------------------------------------------------------------------
    // // //----------------------------------------------GRAYSCALE: We then add prod for 1...10----------------------------------------------
    let polIndex = 12;
    let ptIndex = 4;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range for prod----------------------------------------------
    let polIndex = 12;
    let ptIndex = 27;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range for frac----------------------------------------------

    let polIndex = 11;
    let ptIndex = 27;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());

    // //----------------------------------------------GRAYSCALE: we now add alpha_range||0 for prod----------------------------------------------
    let polIndex = 12;
    let ptIndex = 28;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());

    // // // // //----------------------------------------------GRAYSCALE: we now add alpha_range||0 for frac----------------------------------------------
    let polIndex = 11;
    let ptIndex = 28;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());
    
    // // // //----------------------------------------------GRAYSCALE: we now add alpha_range||1 for prod----------------------------------------------
    let polIndex = 12;
    let ptIndex = 29;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());

    // // // //----------------------------------------------GRAYSCALE: we now add alpha_range||1 for frac----------------------------------------------
    let polIndex = 11;
    let ptIndex = 29;
    evalPolsBig.push(polies2[polIndex].clone());
    evalPointsBig.push(points[ptIndex].clone());
    evalValsBig.push(polies2[polIndex].evaluate(&points[ptIndex]).unwrap());
    evalComsBig.push(coms[polIndex+3].clone());   
    // // // //----------------------------------------------DONE WITH OPENINGS ----------------------------------------------
        
    let openProofs = PCS::multi_open(&pcs_param,&evalPols,&evalPoints,&evalVals,&mut transcript).unwrap();
    let openProofsBig = PCS::multi_open(&pcs_param,&evalPolsBig,&evalPointsBig,&evalValsBig,&mut transcript).unwrap();

    let elapsed_time = now0.elapsed();
    println!("\nTime to do WHOLE PROVER TIME is {:?} seconds\n", elapsed_time.as_millis() as f64 / 1000 as f64);

    println!("\n------------------------------------------\nComputing Proof Sizes");

    let mut vec_multsetProofGray = Vec::new();
    vec_multsetProofGray.push(&multsetProofGray);

    let (elems_bls, elems_256, elems_scalar) = get_proof_size(&ver_param, irredPolyTable[numCols].try_into().unwrap(), 
        &coms, &testDigestRGB, &proofRGB,
        &poly_info_matMult, &multsetProof,
        &vec_multsetProofGray, 
        &evalComs, &evalComsBig,
        &openProofs, &openProofsBig, 
        &evalVals, &evalValsBig,
        &hComs);

    println!("Total Bls12_381 elements: {:?}", elems_bls);
    println!("Total 256 bit elements: {:?}", elems_256);
    println!("Total Bls12_381 scalar field elements: {:?}", elems_scalar);

    let total_bls_bytes = elems_bls * 48;
    let total_256_bytes = elems_256 * 32;
    let total_scalar_bytes = elems_scalar * 32;

    let total_bytes = total_bls_bytes + total_256_bytes + total_scalar_bytes;

    println!("\n## TOTAL PROOF SIZE (Hash + Crop): {:?} Bytes", total_bytes);
    
    println!("\n\n");


    let mut verTranscript =
    <PolyIOP<F> as ProductCheck<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::init_transcript();

    let ver_start: Instant = Instant::now();

    ver(&ver_param, testSize, irredPolyTable[numCols].try_into().unwrap(), numCols, numRows, numCols, 
        coms,testDigestRGB, proofRGB,
        poly_info_matMult, multsetProof, 
        poly_infoProds,
        multsetProofGray, auxGray, 
        evalComs, evalComsBig,
        openProofs, openProofsBig, 
        evalVals, evalValsBig,
        hComs, grayErrCom, hGrayCom, &mut verTranscript);

    let elapsed_ver = ver_start.elapsed();
    println!("Time to do WHOLE VERIFIER is {:?} seconds", elapsed_ver.as_millis() as f64 / 1000 as f64);
}

fn ver<E, PCS>(
    ver_param: &PCS::VerifierParam,
    testSize: usize, 
    primPolyForT: u64,
    nvOrig: usize,
    numRows: usize, 
    numCols: usize, 
    coms: Vec<Commitment<ark_ec::bls12::Bls12<ark_bls12_381::Config>>>,
    hashVal: Vec<Vec<F>>,
    proofRGB: [<PolyIOP<F> as SumCheck<F>>::SumCheckProof;3], 
    poly_infoRGB: [VPAuxInfo<F>;3],
    multsetProof: Vec<<PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::ProductCheckProof>,
    poly_infoProds: Vec<VPAuxInfo<F>>,
    multsetProofGray: <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::ProductCheckProof,
    poly_infoGray: VPAuxInfo<F>,
    evalComs: Vec<<PCS as PolynomialCommitmentScheme<E>>::Commitment>,
    evalComsBig: Vec<<PCS as PolynomialCommitmentScheme<E>>::Commitment>,
    openProofs: BatchProof<ark_ec::bls12::Bls12<ark_bls12_381::Config>, MultilinearKzgPCS<ark_ec::bls12::Bls12<ark_bls12_381::Config>>>,
    openProofsBig:BatchProof<ark_ec::bls12::Bls12<ark_bls12_381::Config>, MultilinearKzgPCS<ark_ec::bls12::Bls12<ark_bls12_381::Config>>>,
    evalVals:Vec<F>,
    evalValsBig:Vec<F>,
    hComs: Vec<Commitment<ark_ec::bls12::Bls12<ark_bls12_381::Config>>>,
    grayErrCom:Commitment<ark_ec::bls12::Bls12<ark_bls12_381::Config>>,
    hGrayCom: Commitment<ark_ec::bls12::Bls12<ark_bls12_381::Config>>,
    transcript: &mut IOPTranscript<E::ScalarField>,
    )
    where
    E: Pairing<ScalarField = F>,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<F>,
        Evaluation = F,
        BatchProof = BatchProof<Bls12_381, MultilinearKzgPCS<Bls12_381>>
    >,          {
    let now = Instant::now();
    let maxVal = 255;
    
    // Initialize transcript
    for i in 0..3{
        transcript.append_serializable_element(b"img(x)", &coms[i]);
    };
    

    // Initialize Randomness
    let mut matrixA = Vec::new();
    for i in 0..128 {
        matrixA.push(ChaCha8Rng::seed_from_u64(i));
    }
    // ======================================================Do collapse of hash matrix======================================================
    let mut frievaldRandVec = Vec::new();

    for i in 0..(1 << numRows) {
        let alpha = transcript.get_and_append_challenge(b"alpha").unwrap();

        frievaldRandVec.push(alpha);
    }
   
    //We make rT*A
    let now = Instant::now();

    let nowJank: Instant = Instant::now();

    let mut rTA = Vec::new();

    for i in 0..(1 << numCols) {
        let mut mySum = F::zero();
        for j in 0..128 {
            mySum += F::rand(&mut matrixA[j]) * frievaldRandVec[j];
        }
        rTA.push(mySum);
    }
    let elapsed_time = nowJank.elapsed();
    println!("KZG: Time to compute rTA is {:?} seconds", elapsed_time.as_millis() as f64 / 1000 as f64);
    // The verifier computes HASH * Frievald
    let mut expectedSumVal = [F::zero(),F::zero(),F::zero()];
    for i in 0..3{
        for j in 0..(1<<numRows){
            expectedSumVal[i] += frievaldRandVec[j]* hashVal[i][j];
        }
    }
    // ======================================================End of collapse of hash matrix==============================================================

    let mut sumCheckForHash = Vec::new();
    for i in 0..3{
        sumCheckForHash.push(<PolyIOP<F> as SumCheck<F>>::verify(expectedSumVal[i], &proofRGB[i], &poly_infoRGB[i], transcript).unwrap());
    }

    println!("Sumchecks for rA I == rh have passed!");
    // Now do sumcheck for range check!
    // These are utilized internally in range check. We'll need them later for when we do point equality checks
    let mut alpha1 = Vec::new(); 

    let mut alpha2 = Vec::new();
    let mut prodCheckSubclaims = Vec::new();
    for i in 0..3{
        alpha1.push(transcript.get_and_append_challenge(b"alpha").unwrap());
        alpha2.push(transcript.get_and_append_challenge(b"alpha").unwrap());
        prodCheckSubclaims.push(<PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::verify(&multsetProof[i],&poly_infoProds[i], transcript).unwrap());
    }
    println!("Sumchecks for rangechecks have passed!");
    // Now do sumcheck for image transformation
    // First get frievald challenge

    for i in 0..3{
        transcript.append_serializable_element(b"hCom(x)", &hComs[i]);
    }

    transcript.append_serializable_element(b"gray(x)", &grayErrCom);

    let alpha1Gray = transcript.get_and_append_challenge(b"alpha").unwrap();
    let alpha2Gray = transcript.get_and_append_challenge(b"alpha").unwrap();
    let prodCheckSubclaimsGray = <PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::verify(&multsetProofGray,&poly_infoGray, transcript).unwrap();

    transcript.append_serializable_element(b"hGrayCom(x)", &hGrayCom);

    // Start of getting eval points
    let mut evalPoints = Vec::new();
    let mut evalPointsBig = Vec::new();


    // ==============================We first create the table image polynomial that the verifer must use as part of the range check!===================================
    let mut embeddedTable: Vec<F> = vec![F::zero(); 1 << nvOrig];
    let mut plusOneTable: Vec<F> = vec![F::zero(); 1 << nvOrig];

    //This takes the coefficients of our poly that aren't the most significant one.
    let galoisRep = (primPolyForT) - (1 << nvOrig);
    
    //This is how big our table is
    let size = 1 << nvOrig;
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

    let polyTable = DenseMultilinearExtension::from_evaluations_vec(nvOrig, embeddedTable.clone());

    let polyPlusOneTable =
        DenseMultilinearExtension::from_evaluations_vec(nvOrig, plusOneTable.clone());
    // ============================== We have finished creating the table =======================================
    let mut points = Vec::new();
    let mut startVals = Vec::new();

    // HASH PREIMAGE POINT
    for i in 0..3{
    points.push(proofRGB[i].point.clone());
    }
    // 0 vector, used for h
    points.push(vec![F::zero(); numCols+1]);
    // 1..10 vector, used for prod 
    let mut final_query = vec![F::one(); numCols+1];
    final_query[0] = F::zero();
    points.push(final_query);
    // Eval for range for image
    for i in 0.. 3{
        let myRand = &multsetProof[i].zero_check_proof.point;
        
        let mut myRandSmall = Vec::new();
        for i in 0..myRand.len()-1{
            myRandSmall.push(myRand[i]);
        }
        points.push(myRandSmall.clone());
        myRandSmall = myRand.clone();

        let galoisRep = irredPolyTable[numCols + 1] - (1 <<( numCols+1));
        let (fiddle, zero, startVal) = galoisifyPt::<F>((numCols+1) as u32, galoisRep, myRand.clone());
        startVals.push(startVal);

        // point 1 for h_{+1}
        points.push(fiddle);
        // point 2 for h_{+1}
        points.push( zero.clone());
        //Rand point used for prod and frac polies 
       
        points.push(myRand.clone());
        // Randpoint but last is 0
        let mut ptRand= Vec::new();
        ptRand.push(F::zero());
        for i in 0..myRand.clone().len()-1{
            ptRand.push(myRand[i]);
        }
        points.push(ptRand.clone());
        // Randpoint but last is 1
        ptRand[0] = F::one();
        points.push(ptRand.clone());    
    }

    // Transformation point for grayscale
    let grayTransformPt = transcript.get_and_append_challenge_vectors(b"alpha", numCols).unwrap();
    
    points.push(grayTransformPt.clone());

    let myRand = &multsetProofGray.zero_check_proof.point;
        
    let mut myRandSmall = Vec::new();
    for i in 0..myRand.len()-1{
        myRandSmall.push(myRand[i]);
    }
    points.push(myRandSmall.clone());
    myRandSmall = myRand.clone();

    let galoisRep = irredPolyTable[numCols + 1] - (1 <<( numCols+1));
    let (fiddle, zero, startVal) = galoisifyPt::<F>((numCols+1) as u32, galoisRep, myRand.clone());
    // point 1 for h_{+1}
    startVals.push(startVal);

    points.push(fiddle);
    // point 2 for h_{+1}
    points.push( zero.clone());
    //Rand point used for prod and frac polies 
    
    points.push(myRand.clone());
    // Randpoint but last is 0
    let mut ptRand= Vec::new();
    ptRand.push(F::zero());
    for i in 0..myRand.clone().len()-1{
        ptRand.push(myRand[i]);
    }
    points.push(ptRand.clone());
    // Randpoint but last is 1
    ptRand[0] = F::one();
    points.push(ptRand.clone());
   
   for i in 0..3{
        // //----------------------------------------------We first add opening for matrixMult for hash.----------------------------------------------    
        let ptIndex = i;
        evalPoints.push(points[ptIndex].clone());
        
        // // //----------------------------------------------We now add alpha_range for image----------------------------------------------
        
        let ptIndex = 5+6*i;        
        evalPoints.push(points[ptIndex].clone());
        // // //WE NOW DO BIG POLYNOMIALS --------------------------------------------------------------------------------------------
         // //----------------------------------------------We now add 0 for h----------------------------------------------
        let ptIndex = 3;
        evalPointsBig.push(points[ptIndex].clone());
        // // //----------------------------------------------We now add alpha_range for h----------------------------------------------

        let ptIndex = 8+6*i;
        evalPointsBig.push(points[ptIndex].clone());
        // // //----------------------------------------------We now add alpha_range Fiddle for h----------------------------------------------

        let ptIndex = 6+6*i;
        evalPointsBig.push(points[ptIndex].clone());

        // // //----------------------------------------------We now add alpha_range 0 for h----------------------------------------------
        let ptIndex = 7+6*i;
        evalPointsBig.push(points[ptIndex].clone());
        // // //----------------------------------------------We then add prod for 1...10----------------------------------------------
        let ptIndex = 4;
        evalPointsBig.push(points[ptIndex].clone());

        // // //----------------------------------------------We now add alpha_range for prod----------------------------------------------
        let ptIndex = 8+6*i;
        evalPointsBig.push(points[ptIndex].clone());

        // // //----------------------------------------------We now add alpha_range for frac----------------------------------------------

        let ptIndex = 8+6*i;
        evalPointsBig.push(points[ptIndex].clone());

        // //----------------------------------------------we now add alpha_range||0 for prod----------------------------------------------
        let ptIndex = 9+6*i;
        evalPointsBig.push(points[ptIndex].clone());

        // // // // //----------------------------------------------we now add alpha_range||0 for frac----------------------------------------------
        let ptIndex = 9+6*i;
        evalPointsBig.push(points[ptIndex].clone());

        // // // //----------------------------------------------we now add alpha_range||1 for prod----------------------------------------------
        let ptIndex = 10+6*i;
        evalPointsBig.push(points[ptIndex].clone());

        // // // //----------------------------------------------we now add alpha_range||1 for frac----------------------------------------------
        let ptIndex = 10+6*i;
        evalPointsBig.push(points[ptIndex].clone());
        // // // //----------------------------------------------we now add transPoint for IMG ----------------------------------------------
        let ptIndex = 23;
        evalPoints.push(points[ptIndex].clone());
    }

    //----------------------------------------------GRAYSCALE: We first add opening for transPoint for GRAY.----------------------------------------------    
    let ptIndex = 23;
    evalPoints.push(points[ptIndex].clone());
    // //----------------------------------------------GRAYSCALE: We now add 0 for h----------------------------------------------
    let ptIndex = 3;
    evalPointsBig.push(points[ptIndex].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range for image----------------------------------------------
    let ptIndex = 24;
    evalPoints.push(points[ptIndex].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range for h----------------------------------------------

    let ptIndex = 27;
    evalPointsBig.push(points[ptIndex].clone());
    // // //----------------------------------------------GRAYSCALE: We now add alpha_range modified0 for h----------------------------------------------

    let ptIndex = 25;
    evalPointsBig.push(points[ptIndex].clone());
    // // //----------------------------------------------GRAYSCALE: We now add alpha_range modified1 for h----------------------------------------------
    let ptIndex = 26;
    evalPointsBig.push(points[ptIndex].clone());
    
    // // //WE NOW DO BIG POLYNOMIALS --------------------------------------------------------------------------------------------
    // // //----------------------------------------------GRAYSCALE: We then add prod for 1...10----------------------------------------------
    let ptIndex = 4;
    evalPointsBig.push(points[ptIndex].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range for prod----------------------------------------------
    let ptIndex = 27;
    evalPointsBig.push(points[ptIndex].clone());

    // // //----------------------------------------------GRAYSCALE: We now add alpha_range for frac----------------------------------------------

    let ptIndex = 27;
    evalPointsBig.push(points[ptIndex].clone());
    // //----------------------------------------------GRAYSCALE: we now add alpha_range||0 for prod----------------------------------------------
    let ptIndex = 28;
    evalPointsBig.push(points[ptIndex].clone());
    
    // // // // //----------------------------------------------GRAYSCALE: we now add alpha_range||0 for frac----------------------------------------------
    let ptIndex = 28;
    evalPointsBig.push(points[ptIndex].clone());
    
    // // // //----------------------------------------------GRAYSCALE: we now add alpha_range||1 for prod----------------------------------------------
    let ptIndex = 29;
    evalPointsBig.push(points[ptIndex].clone());


    // // // //----------------------------------------------GRAYSCALE: we now add alpha_range||1 for frac----------------------------------------------
    let ptIndex = 29;
    evalPointsBig.push(points[ptIndex].clone());
  

    PCS::batch_verify(&ver_param,&evalComs,&evalPoints,&openProofs, transcript).unwrap();

    PCS::batch_verify(&ver_param,&evalComsBig,&evalPointsBig,&openProofsBig, transcript).unwrap();

    // Now ensure that the values generated are in fact correct!
    let mut flag = true;
    let myZero = (openProofs.f_i_eval_at_point_i[1]-openProofs.f_i_eval_at_point_i[1]);
    let myOne = (openProofs.f_i_eval_at_point_i[0]/openProofs.f_i_eval_at_point_i[0]);

    // We need to check that      

    // FOR RANGECHECK; for h, h(0) = 0.
    for i in 0..3{
        flag = flag && (openProofsBig.f_i_eval_at_point_i[11*i] ==myZero);
    }

    // FOR PRODUCT CHECK; for v, v(1,..,1,0)= 1;
    for i in 0..3{
        flag = flag && (openProofsBig.f_i_eval_at_point_i[4+11*i] ==myOne);
    }

    // We now compute the MONSTER VALUE
    for i in 0..3{
        let myRand = &multsetProof[i].zero_check_proof.point;
        let mut myRandSmall = Vec::new();
        for i in 0..myRand.len()-1{
            myRandSmall.push(myRand[i]);
        }
        let lastVal = myRand[myRand.len()-1];
        let imgAtAlphaSmall = openProofs.f_i_eval_at_point_i[1 + i*3];
        let hAtAlphaRange =  openProofsBig.f_i_eval_at_point_i[1+11*i];
        let hAtAlphaRangeFiddle = openProofsBig.f_i_eval_at_point_i[2+11*i];
        let hAtAlphaRange0 =  openProofsBig.f_i_eval_at_point_i[3+11*i];  
        let prodAtAlphaRange = openProofsBig.f_i_eval_at_point_i[5+11*i];
        let fracAtAlphaRange = openProofsBig.f_i_eval_at_point_i[6+11*i];
        let prodAtAlphaRange0 = openProofsBig.f_i_eval_at_point_i[7+11*i];
        let fracAtAlphaRange0 = openProofsBig.f_i_eval_at_point_i[8+11*i];
        let prodAtAlphaRange1 = openProofsBig.f_i_eval_at_point_i[9+11*i];
        let fracAtAlphaRange1 = openProofsBig.f_i_eval_at_point_i[10+11*i];

        // We first compute prod(x) - v(x,0)v(x,1)
        let mut firstHalf = prodAtAlphaRange;
        
        //Inevstigate this and virtual polynomial...
        let myAlpha = myRand[myRand.len()-1]; 
        let vX0 = myAlpha * prodAtAlphaRange0 + (F::one()- myAlpha) * fracAtAlphaRange0;
        let vX1 =  myAlpha * prodAtAlphaRange1 + (F::one()- myAlpha) * fracAtAlphaRange1;
        firstHalf += -vX0 * vX1;

        // We are done creating first half

        // alpha0 + merge(I,T)(X) + alpha1 merge(I,T_{+1})(X)
        let mut f1 = alpha1[i] + ((F::one()- lastVal) * imgAtAlphaSmall + lastVal * (polyTable.evaluate(&myRandSmall).unwrap()));
        f1 += alpha2[i] * ((F::one()- lastVal) * imgAtAlphaSmall + lastVal* polyPlusOneTable.evaluate(&myRandSmall).unwrap());
        // alpha0 + h(X) + alpha1 h_{+1}(X)

        let mut f2 =alpha1[i] + hAtAlphaRange + alpha2[i] * (startVals[i] * hAtAlphaRangeFiddle +  (F::one()-startVals[i])* hAtAlphaRange0) ;
        let mut secondHalf = ( f2 * fracAtAlphaRange - f1);
        secondHalf = secondHalf * prodCheckSubclaims[i].alpha;
        let anticipatedVal = prodCheckSubclaims[i].zero_check_sub_claim.expected_evaluation;

        let finalVal = firstHalf+secondHalf;
        flag = flag && anticipatedVal == finalVal;
    }

    // We now AFFIRM THE ERROR CHECK...
    let redVal = F::from_le_bytes_mod_order(&[30]) * evalVals[2]; 
    let greenVal = F::from_le_bytes_mod_order(&[59]) * evalVals[5];
    let blueVal = F::from_le_bytes_mod_order(&[11]) * evalVals[8];

    let errorVal =  evalVals[9] - F::from_le_bytes_mod_order(&[50]);

    // We now read the grayscaled image and evaluate it at alpha...
    let grayFileName = format!("images/Gray{}.json", testSize);

    let grayImg = load_image(grayFileName);
    let (grayImg, _)  = vec_to_poly(toFieldVec::<F>(&grayImg.R));    
    let grayVal = F::from_le_bytes_mod_order(&[100]) * grayImg.evaluate(&grayTransformPt).unwrap();

    flag = flag && (grayVal == (redVal + greenVal + blueVal + errorVal));

    // We now verify that the "v" polynomial and "h" polynomial for grayscale 
    // evaluated at (1,..,1,0) and (0...0) respectively eval to what we expect;

    // THIS IS FOR h
    flag = flag && openProofsBig.f_i_eval_at_point_i[33] == F::zero();
    // THIS IS FOR PROD
    flag = flag && openProofsBig.f_i_eval_at_point_i[37] == F::one();

    let maxValGray = 100;
    // LAST STEP... VERIFY THAT THE RANGE CHECK HOLDS FOR THE ERROR TERM!
    let mut embeddedTable: Vec<F> = vec![F::zero(); 1 << nvOrig];
    let mut plusOneTable: Vec<F> = vec![F::zero(); 1 << nvOrig];

    //This takes the coefficients of our poly that aren't the most significant one.
    let galoisRep = (primPolyForT) - (1 << nvOrig);
    
    //This is how big our table is
    let size = 1 << nvOrig;
    let mut binaryString: u64 = 1;
    //We create the table by setting index i to g^i(1) where g is our generator.
    for i in 1..(maxValGray as usize + 1) {
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

    let polyTable = DenseMultilinearExtension::from_evaluations_vec(nvOrig, embeddedTable.clone());

    let polyPlusOneTable =
        DenseMultilinearExtension::from_evaluations_vec(nvOrig, plusOneTable.clone());
    // We first verify that h and prod are as expected
    let myRand = &multsetProofGray.zero_check_proof.point;

    let mut myRandSmall = Vec::new();
    for i in 0..myRand.len()-1{
        myRandSmall.push(myRand[i]);
    }
    let lastVal = myRand[myRand.len()-1];
    let imgAtAlphaSmall = openProofs.f_i_eval_at_point_i[10];
    let hAtAlphaRange =  openProofsBig.f_i_eval_at_point_i[34];
    let hAtAlphaRangeFiddle = openProofsBig.f_i_eval_at_point_i[35];
    let hAtAlphaRange0 =  openProofsBig.f_i_eval_at_point_i[36];  
    let prodAtAlphaRange = openProofsBig.f_i_eval_at_point_i[38];
    let fracAtAlphaRange = openProofsBig.f_i_eval_at_point_i[39];
    let prodAtAlphaRange0 = openProofsBig.f_i_eval_at_point_i[40];
    let fracAtAlphaRange0 = openProofsBig.f_i_eval_at_point_i[41];
    let prodAtAlphaRange1 = openProofsBig.f_i_eval_at_point_i[42];
    let fracAtAlphaRange1 = openProofsBig.f_i_eval_at_point_i[43];

    // We first compute prod(x) - v(x,0)v(x,1)
    let mut firstHalf = prodAtAlphaRange;
    
    let myAlpha = myRand[myRand.len()-1]; 
    let vX0 = myAlpha * prodAtAlphaRange0 + (F::one()- myAlpha) * fracAtAlphaRange0;
    let vX1 =  myAlpha * prodAtAlphaRange1 + (F::one()- myAlpha) * fracAtAlphaRange1;
    firstHalf += -vX0 * vX1;

    // alpha0 + merge(I,T)(X) + alpha1 merge(I,T_{+1})(X)
    let mut f1 = alpha1Gray + ((F::one()- lastVal) * imgAtAlphaSmall + lastVal * (polyTable.evaluate(&myRandSmall).unwrap()));
    f1 += alpha2Gray * ((F::one()- lastVal) * imgAtAlphaSmall + lastVal* polyPlusOneTable.evaluate(&myRandSmall).unwrap());
    // alpha0 + h(X) + alpha1 h_{+1}(X)

    let mut f2 =alpha1Gray + hAtAlphaRange + alpha2Gray * (startVals[3] * hAtAlphaRangeFiddle +  (F::one()-startVals[3])* hAtAlphaRange0) ;
    let mut secondHalf = ( f2 * fracAtAlphaRange - f1);
    secondHalf = secondHalf * prodCheckSubclaimsGray.alpha;
    let anticipatedVal = prodCheckSubclaimsGray.zero_check_sub_claim.expected_evaluation;

    let finalVal = firstHalf+secondHalf;
    flag = flag && anticipatedVal == finalVal;

    println!("Verifier passed!: {:?}", flag);

    println!("verifier done!\n");
}

fn get_proof_size<E, PCS>(
    ver_param: &PCS::VerifierParam,
    primPolyForT: u64,
    coms: &Vec<Commitment<ark_ec::bls12::Bls12<ark_bls12_381::Config>>> ,
    hashVal: &Vec<Vec<F>>,
    proofRGB: &[<PolyIOP<F> as SumCheck<F>>::SumCheckProof;3], 
    poly_infoRGB: &[VPAuxInfo<F>;3],
    multsetProof: &Vec<<PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::ProductCheckProof>,
    vec_multsetProofGray: &Vec<&<PolyIOP<E::ScalarField> as ProductCheck<E, PCS>>::ProductCheckProof>,
    evalComs: &Vec<<PCS as PolynomialCommitmentScheme<E>>::Commitment>,
    evalComsBig: &Vec<<PCS as PolynomialCommitmentScheme<E>>::Commitment>,
    openProofs: &BatchProof<ark_ec::bls12::Bls12<ark_bls12_381::Config>, MultilinearKzgPCS<ark_ec::bls12::Bls12<ark_bls12_381::Config>>>,
    openProofsBig: &BatchProof<ark_ec::bls12::Bls12<ark_bls12_381::Config>, MultilinearKzgPCS<ark_ec::bls12::Bls12<ark_bls12_381::Config>>>,
    evalVals: &Vec<F>,
    evalValsBig: &Vec<F>,
    hComs: &Vec<Commitment<ark_ec::bls12::Bls12<ark_bls12_381::Config>>>,
    ) -> (usize, usize, usize)
    where
    E: Pairing<ScalarField = F>,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<F>,
        Evaluation = F,
        BatchProof = BatchProof<Bls12_381, MultilinearKzgPCS<Bls12_381>>
    >,       
    {
    
    let mut total_bls_elems = 0;
    let mut total_256_elems = 0;
    let mut total_scalar_field_elems = 0;

    // add number of commits to total bls elems
    total_bls_elems += coms.len();
    total_bls_elems += hComs.len();
    total_bls_elems += evalComs.len();
    total_bls_elems += evalComsBig.len();

    // add field elems from this hash val
    for hash in hashVal {
        total_256_elems += hash.len();
    }

    // add num of field elems from proofRGB
    for i in 0..3 {
        total_256_elems += proofRGB[i].point.len();

        for pf in proofRGB[i].clone().proofs {
            // how can I determine the length of evaluations???
            total_256_elems += pf.evaluations.len();
        }
    }

    // add field and group elems from multsetProof
    for multsetPf in multsetProof {
        // add 2 group elems (comes from the frac and prod polys in multsetPf)
        total_bls_elems += 2;

        let zero_pf = &multsetPf.zero_check_proof;
        total_256_elems += zero_pf.point.len();

        for pf in zero_pf.clone().proofs {
            // how can I determine the length of evaluations???
            total_256_elems += pf.evaluations.len();
        }
    }

    // add num of field and group elems from multsetProofGray
    let multsetProofGray = &vec_multsetProofGray[0];
    total_bls_elems += 2;
    let zero_pf = &multsetProofGray.zero_check_proof;
    total_256_elems += zero_pf.point.len();

    for pf in zero_pf.clone().proofs {
        total_256_elems += pf.evaluations.len();
    }

    // add number of field elems from the evalVals vectors
    total_256_elems += evalVals.len();
    total_256_elems += evalValsBig.len();

    // add stuff from openProofs
    total_scalar_field_elems += &openProofs.f_i_eval_at_point_i.len();

    total_bls_elems += &openProofs.g_prime_proof.proofs.len();

    total_scalar_field_elems += &openProofs.sum_check_proof.point.len();

    for p_msg in &openProofs.sum_check_proof.proofs {
        total_scalar_field_elems += p_msg.evaluations.len();
    }

    // add stuff from openProofsBig
    total_scalar_field_elems += &openProofsBig.f_i_eval_at_point_i.len();

    total_bls_elems += &openProofsBig.g_prime_proof.proofs.len();

    total_scalar_field_elems += &openProofsBig.sum_check_proof.point.len();
    
    for p_msg in &openProofsBig.sum_check_proof.proofs {
        total_scalar_field_elems += p_msg.evaluations.len();
    }


    return (total_bls_elems, total_256_elems, total_scalar_field_elems)
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
        println!("Full System Grayscale, HyperVerITAS PST. Size: 2^{:?}\n", i);
        let _res = run_full_gray_kzg(i);
        println!("-----------------------------------------------------------------------");
    }
}