#![allow(warnings)]

mod iop_brakedown_256;
use iop_brakedown_256::*;

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
        hash::Blake2s256,
        new_fields::Mersenne127,
        code::{Brakedown, BrakedownSpec3, BrakedownSpec6},
        expression::{CommonPolynomial, Expression, Query, Rotation}, 
        arithmetic::{BatchInvert, BooleanHypercube, Field as myField}, 
        transcript::{Blake2s256Transcript, FiatShamirTranscript, FieldTranscript, FieldTranscriptRead, FieldTranscriptWrite, InMemoryTranscript, TranscriptWrite},
    },
    halo2_curves::bn256::Fr as F,
};


type Pcs = MultilinearBrakedown<F, Blake2s256, BrakedownSpec6>;
type VT = FiatShamirTranscript<Blake2s256, std::io::Cursor<Vec<u8>>>;


const irredPolyTable: &[u32] = &[
    0, 0, 7, 11, 19, 37, 67, 131, 285, 529, 1033, 2053, 4179, 8219, 16707, 32771, 69643, 131081,
    262273, 524327, 1048585, 2097157, 4194307, 8388641, 16777351, 33554441, 67108935,
];


pub fn eq_eval(x: &[F], y: &[F]) -> F {
    let mut res = F::ONE;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let xi_yi = xi * yi;
        res *= xi_yi + xi_yi - xi - yi + F::ONE;
    }
    return(res)
}

pub fn matSparseMultVec(
    numRows: usize,
    numCols: usize,
    sprseRep: &[Vec<(usize, F)>],
    r: &[F],
) -> Vec<F> {
    let mut Ar = Vec::new();
    for i in 0..numRows {
        let mut mySum = F::ZERO;
        for j in 0..sprseRep[i].len() {
            mySum += sprseRep[i][j].1 * r[sprseRep[i][j].0];
        }
        Ar.push(mySum);
    }
    return Ar;
}

fn makePtsFullGray(numCols: usize, vals: Vec<Vec<F>>)-> Vec<Vec<F>>{
    let mut points = Vec::new();

    // Original hash pre-image point
    let mut origPt: Vec<F> = vals[0].clone();
    origPt.push(F::ZERO);
    points.push(origPt.clone());
    
    // 0 vector, used for h
    let mut pt0: Vec<F> = vec![F::ZERO; numCols+1];
    points.push(pt0.clone());
    // 1..10 vector, used for prod 
    let mut final_query = vec![F::ONE; numCols+1];
    final_query[0] = F::ZERO;
    points.push(final_query);
    // Eval for range for image
    for i in 0.. 4{
        let mut myRand = vals[1+i].clone();
        myRand[numCols] = F::ZERO;
        points.push(myRand.clone());

        let mut myRand = vals[1+i].clone();

        // point 1 for h_{+1}
        let galoisRep = irredPolyTable[numCols + 1] - (1 <<( numCols+1));
        let (fiddle, zero, startVal) = galoisifyPt((numCols+1) as u32, galoisRep, myRand.clone());

        points.push(fiddle);
        // point 2 for h_{+1}
        points.push(zero);

        //Rand point used for prod and frac polies 
        points.push(myRand.clone());
    
        // Randpoint but last is 0
        let mut ptRand= Vec::new();
        ptRand.push(F::ZERO);
        for i in 0..myRand.clone().len()-1{
            ptRand.push(myRand[i]);
        }
        points.push(ptRand.clone());
        // Randpoint but last is 1
        ptRand[0] = F::ONE;
        points.push(ptRand.clone()); 
    }
    let mut transPointOrig = vals[5].clone();
    points.push(transPointOrig);
    return points;
}


fn hashPreimageProve(
    pp: <Pcs as PolynomialCommitmentScheme<F>>::ProverParam,
    numCols: usize,
    numRows: usize,
    RGBEvals: [Vec<F>;3],
    RBGEvalsInt: [Vec<usize>;3],
    transcript: &mut (impl TranscriptWrite<<Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F> + InMemoryTranscript),
) -> (
    Vec<MultilinearBrakedownCommitment<F, Blake2s256>>,
    Vec<F>,
    Vec<Vec<F>>,
    Vec<MultilinearPolynomial<F>>,
    [Vec<F>;1],
    Vec<MultilinearBrakedownCommitment<F, Blake2s256>>,
    Vec<MultilinearPolynomial<F>>,
    Vec<Vec<F>>,
){
    //We assume we use the randomness matrix.
    let mut rng = test_rng();

    let mut matrixA = Vec::new();
    for i in 0..128 {
        matrixA.push(ChaCha8Rng::seed_from_u64(i));
    }

    //We are given the image polynomial
    let mut imgPolies: Vec<MultilinearPolynomial<F>> = Vec::new();
    for i in 0..3{
        let mut padded = RGBEvals[i].clone();
        padded.append(&mut vec![F::ZERO; 1<< numCols]);
        imgPolies.push(MultilinearPolynomial::new(padded));
    }
   
    let imgComs = Pcs::batch_commit_and_write(&pp, &imgPolies, transcript);
    let mut imgPolies: Vec<MultilinearPolynomial<F>> = Vec::new();
    imgPolies.push(MultilinearPolynomial::<F>::new(RGBEvals[0].clone()));
    imgPolies.push(MultilinearPolynomial::<F>::new(RGBEvals[1].clone()));
    imgPolies.push(MultilinearPolynomial::<F>::new(RGBEvals[2].clone()));

    //We make Frievald random vec
    let frievaldRandVec = transcript.squeeze_challenges(1 << numRows);

    //We make rT*A 
    let mut rTA = Vec::new();

    for i in 0..(1 << numCols) {
        let mut mySum = F::ZERO;
        for j in 0..128 {
            mySum += F::random(&mut matrixA[j]) * frievaldRandVec[j];
        }
        rTA.push(mySum);
    }

    let rTAPoly = MultilinearPolynomial::<F>::new(rTA.clone());

    //We run the sumcheck on rTA * I
    let poly_0 = Expression::<F>::Polynomial(Query::new(0, Rotation::cur()));
    let poly_1 = Expression::<F>::Polynomial(Query::new(1, Rotation::cur()));
    let poly_2 = Expression::<F>::Polynomial(Query::new(2, Rotation::cur()));
    let poly_3 = Expression::<F>::Polynomial(Query::new(3, Rotation::cur()));

    let alpha_1 = transcript.squeeze_challenge();
    let alpha_2 = transcript.squeeze_challenge();

    let prod = poly_0.clone()  * poly_1 + 
                                                           Expression::Constant(alpha_1) * poly_0.clone() * poly_2 + 
                                                           Expression::Constant(alpha_2) * poly_0.clone() * poly_3;

    let polys = vec![rTAPoly.clone(), imgPolies[0].clone(), imgPolies[1].clone(), imgPolies[2].clone()];

    let challenges = vec![transcript.squeeze_challenge()];
    let rand_vector = transcript.squeeze_challenges(numCols);
    let ys = [rand_vector.clone()];

    let mut my_sum_0 = F::ZERO;
    let mut my_sum_1 = F::ZERO;
    let mut my_sum_2 = F::ZERO;

    let rta_evals = rTAPoly.evals();
    let img0_evals = imgPolies[0].evals();
    let img1_evals = imgPolies[1].evals();
    let img2_evals = imgPolies[2].evals();
    for i in (0..rta_evals.len()){
        my_sum_0 += rta_evals[i] * img0_evals[i];
        my_sum_1 += rta_evals[i] * img1_evals[i];
        my_sum_2 += rta_evals[i] * img2_evals[i];
    }
    let my_sum: F = my_sum_0 + alpha_1 * my_sum_1 + alpha_2 * my_sum_2;

    let proof_mm = 
        <ClassicSumCheck<EvaluationsProver<F>>>::prove(&(), numCols, VirtualPolynomial::new(&prod, &polys, &challenges, &ys), my_sum, transcript).unwrap();
    
    let mut mySum = F::ZERO;

    //We run range check on image
    let mut proofRanges= Vec::new();
    let (mut exp_outs, mut poly_outs, mut chall_outs, mut ys_outs, mut com_outs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..3{
        let mut hTable = vec![0; 257];
        for j in 0..RBGEvalsInt[0].len(){
            hTable[RBGEvalsInt[i][j]] += 1;
        }
        
        let (mut exp_out, mut poly_out, mut chall_out, mut ys_out, mut com_out)= range_checkProverIOP(
            pp.clone(),
            numCols,
            256.try_into().unwrap(),
            hTable,
            imgPolies[i].clone(),
            irredPolyTable[numCols].try_into().unwrap(),
            irredPolyTable[numCols+1].try_into().unwrap(),
            transcript,
            0,
        );

        let proof_range = 
            <ClassicSumCheck<EvaluationsProver<F>>>::prove(&(), numCols+1, VirtualPolynomial::new(&exp_out.clone(), &poly_out.clone(), &chall_out.clone(), &[ys_out.clone()]), F::ZERO, transcript).unwrap();
        
        exp_outs.push(exp_out);
        poly_outs.append(&mut poly_out);
        chall_outs.append(&mut chall_out);
        ys_outs.push(ys_out);
        com_outs.append(&mut com_out);
        proofRanges.push(proof_range.0.clone());
    }

    let range_exp = exp_outs[0].clone();

    return (imgComs.unwrap(), proof_mm.0, proofRanges, imgPolies, ys, com_outs, poly_outs, ys_outs);
}

fn setup(input_size: usize) -> (<Pcs as PolynomialCommitmentScheme<F>>::ProverParam, <Pcs as PolynomialCommitmentScheme<F>>::VerifierParam, Vec<Vec<F>>){
    println!("\nstarting setup");
    let mut rng = test_rng();

    let poly_vars = input_size + 1;

    // param setup
    let (pp, vp) = {
        let poly_size = 1 << (poly_vars);
        let param = Pcs::setup(poly_size, 4, &mut rng).unwrap();
        Pcs::trim(&param, poly_size, 4).unwrap()
    };

    // load image for given input size
    let fileName = format!("images/Timings{}.json", input_size);
    let origImg = load_image(&fileName);

    // splitting image into channels
    let mut RGBEvals =
        [fieldVec::<F>(&origImg.R.iter().map(|&x| x as u64).collect::<Vec<_>>()),
         fieldVec::<F>(&origImg.G.iter().map(|&x| x as u64).collect::<Vec<_>>()),
         fieldVec::<F>(&origImg.B.iter().map(|&x| x as u64).collect::<Vec<_>>()),];

    // creating the hash for the image channels
    let mut digestRGB = Vec::new();
    for k in 0..3{
        // creating hashing matrix
        let mut matrixA = Vec::new();
        for i in 0..128 {
            matrixA.push(ChaCha8Rng::seed_from_u64(i));
        }

        // now do the hashing
        let mut digest = Vec::new();
        for i in 0..128 {
            let mut mySum = F::ZERO;
            for j in 0..(1 << input_size) {
                mySum += F::random(&mut matrixA[i]) * RGBEvals[k][j];
            }
            digest.push(mySum);
        }
        digestRGB.push(digest);
    }

    println!("setup done!\n");
    return (pp, vp, digestRGB)
}

fn prove(pp: <Pcs as PolynomialCommitmentScheme<F>>::ProverParam, input_size: usize, numRows: usize, numCols: usize) 
 -> (Vec<Vec<Evaluation<F>>>, Vec<Vec<Evaluation<F>>>, Vec<Vec<Evaluation<F>>>, Vec<Vec<Evaluation<F>>>, 
     (impl (TranscriptWrite<<Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>) + InMemoryTranscript)) 
{
    println!("starting prover");
    let length = input_size+1;
    
    let mut transcript = Blake2s256Transcript::new(());

    let fileName = format!("images/Timings{}.json", input_size);
    let origImg = load_image(&fileName);

    let mut RGBEvals =
        [fieldVec::<F>(&origImg.R.iter().map(|&x| x as u64).collect::<Vec<_>>()),
         fieldVec::<F>(&origImg.G.iter().map(|&x| x as u64).collect::<Vec<_>>()),
         fieldVec::<F>(&origImg.B.iter().map(|&x| x as u64).collect::<Vec<_>>()),];

    let grayFileName = format!("images/Gray{}.json", input_size);
    let grayImg = load_image(grayFileName);

    let mut r_chan: Vec<usize> = origImg.R.iter().map(|x| (*x).into()).collect();
    let mut g_chan: Vec<usize> = origImg.G.iter().map(|x| (*x).into()).collect();
    let mut b_chan: Vec<usize> = origImg.B.iter().map(|x| (*x).into()).collect();

    let mut gray_chan: Vec<usize> = grayImg.R.iter().map(|x| (*x).into()).collect();
        
    let mut grayError = Vec::new();
    let mut grayErrorAsInt = Vec::new();
    for i in 0..1<<numCols{
        let pushVal = 50+100*gray_chan[i]-(30*r_chan[i]+59*g_chan[i]+11*b_chan[i]);
        grayErrorAsInt.push(pushVal);
        grayError.push(F::from(grayErrorAsInt[i] as u64));
    }

    let grayErrPoly = MultilinearPolynomial::<F>::new(grayError.clone());

    let mut padded = grayErrPoly.clone().evals().to_vec();
    padded.append(&mut vec![F::ZERO; 1<< numCols]);

    let paddedPoly = MultilinearPolynomial::new(padded);
    let mut transcript = Blake2s256Transcript::new(());
    let grayCom =(&mut Pcs::batch_commit_and_write(&pp, &[paddedPoly.clone()], &mut transcript).unwrap());

    let (imgComs, mmChall, rangeChalls, imgPolies,imgYs, com_outs, poly_outs, ys_outs) =
        hashPreimageProve(
            pp.clone(),
            numCols,
            numRows,
            RGBEvals,
            [r_chan, g_chan, b_chan],
            &mut transcript,
        );

    let mut Polies = Vec::new();
    
    for i in 0..3{
        let mut padded = imgPolies[i].clone().evals().to_vec();
        padded.append(&mut vec![F::ZERO; 1<< numCols]);
        Polies.push(MultilinearPolynomial::new(padded));
    }
    Polies.push(paddedPoly.clone());
    for i in 0..3{
        Polies.push(poly_outs[7*i+6].clone());
        Polies.push(poly_outs[7*i].clone());
        Polies.push(poly_outs[7*i+1].clone());
    }

    let mut PolyComs = imgComs.clone();
    PolyComs.append(grayCom);

    for i in 0..3{
        PolyComs.push(com_outs[3*i+2].clone());
        PolyComs.push(com_outs[3*i].clone());
        PolyComs.push(com_outs[3*i+1].clone());
    }

    let mut hTable = vec![0; 257];
    for i in 0..grayErrorAsInt.len(){
        hTable[grayErrorAsInt[i]] += 1;
    }
    let (mut exp_out, mut poly_out, mut chall_out, mut ys_out, mut com_out)= range_checkProverIOP(
        pp.clone(),
        numCols,
        256.try_into().unwrap(),
        hTable,
        grayErrPoly.clone(),
        irredPolyTable[numCols].try_into().unwrap(),
        irredPolyTable[numCols+1].try_into().unwrap(),
        &mut transcript,
        0,
    );
    
    Polies.push(poly_out[6].clone());
    Polies.push(poly_out[0].clone());
    Polies.push(poly_out[1].clone());
    
    PolyComs.push(com_out[2].clone());
    PolyComs.push(com_out[0].clone());
    PolyComs.push(com_out[1].clone());
    let proof_range = 
        <ClassicSumCheck<EvaluationsProver<F>>>::prove(&(), numCols+1, VirtualPolynomial::new(&exp_out.clone(), &poly_out.clone(), &chall_out.clone(), &[ys_out.clone()]), F::ZERO, &mut transcript).unwrap();
    
    let mut polynomials = Polies;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    let mut coms = PolyComs;
  
    let mut my_alphas = Vec::new();
    my_alphas.push(mmChall.clone());
    my_alphas.push(rangeChalls[0].clone());
    my_alphas.push(rangeChalls[1].clone());
    my_alphas.push(rangeChalls[2].clone());
    // Grayscale error challenge
    my_alphas.push(proof_range.0.clone());
    
    let mut grayPt = transcript.squeeze_challenges(numCols);
    grayPt.push(F::ZERO);
    my_alphas.push(grayPt);
    let points = makePtsFullGray(numCols, my_alphas);

    let mut hcoms_vec = Vec::new();
    let mut hpolys_vec = Vec::new();
    let mut hpoints_vec = Vec::new();
    let mut hevals_vec = Vec::new();

    let mut fraccoms_vec = Vec::new();
    let mut fracpolys_vec = Vec::new();
    let mut fracpoints_vec = Vec::new();
    let mut fracevals_vec = Vec::new();

    let mut prodcoms_vec = Vec::new();
    let mut prodpolys_vec = Vec::new();
    let mut prodpoints_vec = Vec::new();
    let mut prodevals_vec = Vec::new();

    let mut imgcoms_vec = Vec::new();
    let mut imgpolys_vec = Vec::new();
    let mut imgpoints_vec = Vec::new();
    let mut imgevals_vec = Vec::new();

    for i in 0..4{
        let hIndex = 4+i*3;
        let prodIndex =6+i*3;
        let fracIndex = 5+i*3;
        // This represents image or grayscale
        let polyIndex = i;

        // MAKE H
        let mut hcom_0 = Vec::new();
        hcom_0.push(&coms[hIndex]);
        hcom_0.push(&coms[hIndex]);
        hcom_0.push(&coms[hIndex]);
        hcom_0.push(&coms[hIndex]);

        let mut hpoly_0 = Vec::new();
        hpoly_0.push(&polynomials[hIndex]);
        hpoly_0.push(&polynomials[hIndex]);
        hpoly_0.push(&polynomials[hIndex]);
        hpoly_0.push(&polynomials[hIndex]);

        let mut hpoints_0 = Vec::new();
        hpoints_0.push(points[1].clone());
        hpoints_0.push(points[6+i*6].clone());
        hpoints_0.push(points[4+i*6].clone());
        hpoints_0.push(points[5+i*6].clone());

        let mut hevals_0 = Vec::new();
        hevals_0.push(Evaluation::new(
            0,
            0,
            F::ZERO,
        ));
        hevals_0.push(Evaluation::new(
            1,
            1,
            hpoly_0[1].evaluate(&hpoints_0[1]),
        ));
        hevals_0.push(Evaluation::new(
            2,
            2,
            hpoly_0[2].evaluate(&hpoints_0[2]),
        ));
        hevals_0.push(Evaluation::new(
            3,
            3,
            hpoly_0[3].evaluate(&hpoints_0[3]),
        ));

        hcoms_vec.push(hcom_0);
        hpolys_vec.push(hpoly_0);
        hpoints_vec.push(hpoints_0);
        hevals_vec.push(hevals_0);

        // make FRAC
        let mut fraccom_0 = Vec::new();
        fraccom_0.push(&coms[fracIndex]);
        fraccom_0.push(&coms[fracIndex]);
        fraccom_0.push(&coms[fracIndex]);

        let mut fracpoly_0 = Vec::new();
        fracpoly_0.push(&polynomials[fracIndex]);
        fracpoly_0.push(&polynomials[fracIndex]);
        fracpoly_0.push(&polynomials[fracIndex]);

        let mut fracpoints_0 = Vec::new();
        fracpoints_0.push(points[6+i*6].clone());
        fracpoints_0.push(points[7+i*6].clone());
        fracpoints_0.push(points[8+i*6].clone());

        let mut fracevals_0 = Vec::new();
        fracevals_0.push(Evaluation::new(
            0,
            0,
            fracpoly_0[0].evaluate(&fracpoints_0[0]),
        ));
        fracevals_0.push(Evaluation::new(
            1,
            1,
            fracpoly_0[1].evaluate(&fracpoints_0[1]),
        ));
        fracevals_0.push(Evaluation::new(
            2,
            2,
            fracpoly_0[2].evaluate(&fracpoints_0[2]),
        ));

        fraccoms_vec.push(fraccom_0);
        fracpolys_vec.push(fracpoly_0);
        fracpoints_vec.push(fracpoints_0);
        fracevals_vec.push(fracevals_0);

        // make PROD
        let mut prodcom_0 = Vec::new();
        prodcom_0.push(&coms[prodIndex]);
        prodcom_0.push(&coms[prodIndex]);
        prodcom_0.push(&coms[prodIndex]);
        prodcom_0.push(&coms[prodIndex]);

        let mut prodpoly_0 = Vec::new();
        prodpoly_0.push(&polynomials[prodIndex]);
        prodpoly_0.push(&polynomials[prodIndex]);
        prodpoly_0.push(&polynomials[prodIndex]);
        prodpoly_0.push(&polynomials[prodIndex]);

        let mut prodpoints_0 = Vec::new();
        prodpoints_0.push(points[2].clone());
        prodpoints_0.push(points[6+i*6].clone());
        prodpoints_0.push(points[7+i*6].clone());
        prodpoints_0.push(points[8+i*6].clone());

        let mut prodevals_0 = Vec::new();
        prodevals_0.push(Evaluation::new(
            0,
            0,
            F::ONE,
        ));
        prodevals_0.push(Evaluation::new(
            1,
            1,
            prodpoly_0[1].evaluate(&prodpoints_0[1]),
        ));
        prodevals_0.push(Evaluation::new(
            2,
            2,
            prodpoly_0[2].evaluate(&prodpoints_0[2]),
        ));
        prodevals_0.push(Evaluation::new(
            3,
            3,
            prodpoly_0[3].evaluate(&prodpoints_0[3]),
        ));

        prodcoms_vec.push(prodcom_0);
        prodpolys_vec.push(prodpoly_0);
        prodpoints_vec.push(prodpoints_0);
        prodevals_vec.push(prodevals_0);

        // make IMG
        let mut imgcom_0 = Vec::new();
        if i < 3{
            imgcom_0.push(&coms[polyIndex]);
        }
        imgcom_0.push(&coms[polyIndex]);
        imgcom_0.push(&coms[polyIndex]);

        let mut imgpoly_0 = Vec::new();
        if i < 3 {
            imgpoly_0.push(&polynomials[polyIndex]);
        }
        imgpoly_0.push(&polynomials[polyIndex]);
        imgpoly_0.push(&polynomials[polyIndex]);

        let mut imgpoints_0 = Vec::new();
        if i < 3{
            imgpoints_0.push(points[0].clone());
        }
        imgpoints_0.push(points[3+i*6].clone());
        imgpoints_0.push(points[27].clone());

        let mut imgevals_0 = Vec::new();
        imgevals_0.push(Evaluation::new(
            0,
            0,
            imgpoly_0[0].evaluate(&imgpoints_0[0]),
        ));
        imgevals_0.push(Evaluation::new(
            1,
            1,
            imgpoly_0[1].evaluate(&imgpoints_0[1]),
        ));

        if i < 3 {
            imgevals_0.push(Evaluation::new(
                2,
                2,
                imgpoly_0[2].evaluate(&imgpoints_0[2]),
            ));
        }

        imgcoms_vec.push(imgcom_0);
        imgpolys_vec.push(imgpoly_0);
        imgpoints_vec.push(imgpoints_0);
        imgevals_vec.push(imgevals_0);

    }

    for i in 0..4{

        transcript.write_field_elements(hevals_vec[i].iter().map(Evaluation::value)).unwrap();
        batch_open_one::<F, Pcs>(
            &pp,
            length,
            hpolys_vec[i].clone(),
            hcoms_vec[i].clone(),
            &hpoints_vec[i],
            &hevals_vec[i],
            &mut transcript,
        ).unwrap();

        transcript.write_field_elements(fracevals_vec[i].iter().map(Evaluation::value)).unwrap();
        batch_open_one::<F, Pcs>(
            &pp,
            length,
            fracpolys_vec[i].clone(),
            fraccoms_vec[i].clone(),
            &fracpoints_vec[i],
            &fracevals_vec[i],
            &mut transcript,
        ).unwrap();

        transcript.write_field_elements(prodevals_vec[i].iter().map(Evaluation::value)).unwrap();
        batch_open_one::<F, Pcs>(
            &pp,
            length,
            prodpolys_vec[i].clone(),
            prodcoms_vec[i].clone(),
            &prodpoints_vec[i],
            &prodevals_vec[i],
            &mut transcript,
        ).unwrap();

        transcript.write_field_elements(imgevals_vec[i].iter().map(Evaluation::value)).unwrap();
        batch_open_one::<F, Pcs>(
            &pp,
            length,
            imgpolys_vec[i].clone(),
            imgcoms_vec[i].clone(),
            &imgpoints_vec[i],
            &imgevals_vec[i],
            &mut transcript,
        ).unwrap();
    }


    return (hevals_vec, fracevals_vec, prodevals_vec, imgevals_vec, transcript)
}

fn verify(
    vp: <Pcs as PolynomialCommitmentScheme<F>>::VerifierParam, 
    numRows: usize,
    numCols: usize,
    hevals_vec: Vec<Vec<Evaluation<F>>>,
    fracevals_vec: Vec<Vec<Evaluation<F>>>,
    prodevals_vec: Vec<Vec<Evaluation<F>>>, 
    imgevals_vec: Vec<Vec<Evaluation<F>>>,
    cameraHash:Vec<Vec<F>>,
    transcript:  (impl (TranscriptWrite<<Pcs as PolynomialCommitmentScheme<F>>::CommitmentChunk, F>) + InMemoryTranscript)) {

    println!("\nstarting verifier");

    // Initialization of commits for opening purposes probably
    let mut commits = Vec::new();
    let mut my_alphas = Vec::new();
    //Initialization of file
    let fileName = format!("images/Gray{}.json", numCols);
    let grayImg = load_image(&fileName);

    // Doing transcript init stuff
    let trans_pf = transcript.into_proof();

    println!("PROOF SIZE: {:?} bytes", trans_pf.len());

    let mut ver_transcript = Blake2s256Transcript::from_proof((), trans_pf.as_slice());
    
    // Append image coms
    let comGray = &mut Pcs::read_commitments(&vp, 1, &mut ver_transcript).unwrap();
    commits.append( &mut Pcs::read_commitments(&vp, 3, &mut ver_transcript).unwrap());
    commits.append(comGray);

    // Squeeze RTA Challenge(Frievald)
    let frievaldRandVecrTA = <VT as FieldTranscript<F>>::squeeze_challenges(&mut ver_transcript,  1<<numRows);

    // Squeeze batching sumcheck vals, alpha1, alpha2
    let alpha_1Hash= <VT as FieldTranscript<F>>::squeeze_challenge(&mut ver_transcript);
    let alpha_2Hash= <VT as FieldTranscript<F>>::squeeze_challenge(&mut ver_transcript);

    // Squeeze challenges for sumcheck
    let challenges: Vec<F> = vec![ <VT as FieldTranscript<F>>::squeeze_challenge(&mut ver_transcript)];
    
    // Squeeze rand_vec for sumcheck
    let rand_vector = <VT as FieldTranscript<F>>::squeeze_challenges(&mut ver_transcript,  numCols);
    // Verify sumcheck

    let mut mySumVals = [F::ZERO,F::ZERO,F::ZERO];
    for i in 0..3{
        for j in 0..1<<numRows{
            mySumVals[i] += frievaldRandVecrTA[j] * cameraHash[i][j] 
        }
    } 

    let mySum = mySumVals[0] + alpha_1Hash*mySumVals[1] + alpha_2Hash * mySumVals[2];   

    let verResCameraHash =  ClassicSumCheck::<EvaluationsProver<F>>::verify(&(), numCols, 2, mySum, &mut ver_transcript).unwrap();
    my_alphas.push(verResCameraHash.clone().1);

    // Done with camera hash part; moving on to range check
    let mut alpha1 = Vec::new();
    let mut alpha2 = Vec::new();
    let mut verResRangeRGB = Vec::new();
    let mut betas = Vec::new();
    let mut maybeChallengeVecs = Vec::new();

    for i in 0..4{
        // Append h table com
        commits.append( &mut Pcs::read_commitments(&vp, 1, &mut ver_transcript).unwrap());
        // Get alpha for the multset check
        alpha1.push(<VT as FieldTranscript<F>>::squeeze_challenge(&mut ver_transcript));
        alpha2.push( <VT as FieldTranscript<F>>::squeeze_challenge(&mut ver_transcript));
        // Append frac then prod coms
        commits.append( &mut Pcs::read_commitments(&vp, 2, &mut ver_transcript).unwrap());
        // Squeeze beta?
        betas.push(<VT as FieldTranscript<F>>::squeeze_challenge(&mut ver_transcript));

        //Squeeze challenges and rand_vector for sumcheck 
        let challenges: Vec<F> = vec![ <VT as FieldTranscript<F>>::squeeze_challenge(&mut ver_transcript)];
        
        let rand_vector = <VT as FieldTranscript<F>>::squeeze_challenges(&mut ver_transcript,  numCols+1);
        // Prove the range sumcheck!
        verResRangeRGB.push(ClassicSumCheck::<EvaluationsProver<F>>::verify(&(), numCols+1, 3, F::ZERO, &mut ver_transcript).unwrap());
        my_alphas.push(verResRangeRGB[i].1.clone());
        maybeChallengeVecs.push(rand_vector);
    }

    let mut imgTrans = <VT as FieldTranscript<F>>::squeeze_challenges(&mut ver_transcript, numCols);
    imgTrans.push(F::ZERO);
    my_alphas.push(imgTrans);
    let points = makePtsFullGray(numCols, my_alphas.clone());

    let mut hpoints_vec = Vec::new();
    let mut hevals_vec2 = Vec::new();

    let mut fracpoints_vec = Vec::new();
    let mut fracevals_vec2 = Vec::new();

    let mut prodpoints_vec = Vec::new();
    let mut prodevals_vec2 = Vec::new();

    let mut imgpoints_vec = Vec::new();
    let mut imgevals_vec2 = Vec::new();

    for i in 0..4{
        let hIndex = 4+i*3;
        let prodIndex =6+i*3;
        let fracIndex = 5+i*3;
        // This represents image or grayscale
        let polyIndex = i;

        // MAKE H
        let mut hpoints_0 = Vec::new();
        hpoints_0.push(points[1].clone());
        hpoints_0.push(points[6+i*6].clone());
        hpoints_0.push(points[4+i*6].clone());
        hpoints_0.push(points[5+i*6].clone());

        hpoints_vec.push(hpoints_0.clone());

        let h_evals: Vec<F> = ver_transcript.read_field_elements(hevals_vec[i].len()).unwrap();
        let mut hevals2= Vec::new();
        
        for j in 0..hevals_vec[i].len(){
            let mut newEval = hevals_vec[i][j].clone();
            newEval.value = h_evals[j];
            hevals2.push(newEval);
        }

        hevals_vec2.push(h_evals.clone());
        
        batch_verify_one::<F, Pcs>(
            &vp,
            numCols+1,
            commits[hIndex].clone(),
            &hpoints_0,
            &hevals2,
            &mut ver_transcript,
        ).unwrap();


        // make FRAC
        let mut fracpoints_0 = Vec::new();
        fracpoints_0.push(points[6+i*6].clone());
        fracpoints_0.push(points[7+i*6].clone());
        fracpoints_0.push(points[8+i*6].clone());

        fracpoints_vec.push(fracpoints_0.clone());

        let frac_evals: Vec<F> = ver_transcript.read_field_elements(fracevals_vec[i].len()).unwrap();
        let mut fracevals2= Vec::new();
        
        for j in 0..fracevals_vec[i].len(){
            let mut newEval = fracevals_vec[i][j].clone();
            newEval.value = frac_evals[j];
            fracevals2.push(newEval);
        }
        
        fracevals_vec2.push(frac_evals.clone());

        batch_verify_one::<F, Pcs>(
            &vp,
            numCols+1,
            commits[fracIndex].clone(),
            &fracpoints_0,
            &fracevals2,
            &mut ver_transcript,
        ).unwrap();


        // make PROD
        let mut prodpoints_0 = Vec::new();
        prodpoints_0.push(points[2].clone());
        prodpoints_0.push(points[6+i*6].clone());
        prodpoints_0.push(points[7+i*6].clone());
        prodpoints_0.push(points[8+i*6].clone());

        prodpoints_vec.push(prodpoints_0.clone());

        let prod_evals: Vec<F> = ver_transcript.read_field_elements(prodevals_vec[i].len()).unwrap();
        let mut prodevals2= Vec::new();
        
        for j in 0..prodevals_vec[i].len(){
            let mut newEval = prodevals_vec[i][j].clone();
            newEval.value = prod_evals[j];
            prodevals2.push(newEval);
        }
        
        prodevals_vec2.push(prod_evals.clone());

        batch_verify_one::<F, Pcs>(
            &vp,
            numCols+1,
            commits[prodIndex].clone(),
            &prodpoints_0,
            &prodevals2,
            &mut ver_transcript,
        ).unwrap();


        // make IMG
        let mut imgpoints_0 = Vec::new();
        if i < 3{
            imgpoints_0.push(points[0].clone());
        }
        imgpoints_0.push(points[3+i*6].clone());
        imgpoints_0.push(points[27].clone());

        imgpoints_vec.push(imgpoints_0.clone());

        let img_evals: Vec<F> = ver_transcript.read_field_elements(imgevals_vec[i].len()).unwrap();
        let mut imgevals2= Vec::new();
        
        for j in 0..imgevals_vec[i].len(){
            let mut newEval = imgevals_vec[i][j].clone();
            newEval.value = img_evals[j];
            imgevals2.push(newEval);
        }
        
        imgevals_vec2.push(img_evals.clone());

        batch_verify_one::<F, Pcs>(
            &vp,
            numCols+1,
            commits[polyIndex].clone(),
            &imgpoints_0,
            &imgevals2,
            &mut ver_transcript,
        ).unwrap();
    }

    // We have done all the opening proofs. Now it's JUST point equality.

    // We compute rTA
    let mut matrixA = Vec::new();
    for i in 0..128 {
        matrixA.push(ChaCha8Rng::seed_from_u64(i));
    }
    let mut rTA = Vec::new();
    for i in 0..(1 << numCols) {
        let mut mySum = F::ZERO;
        for j in 0..128 {
            mySum += F::random(&mut matrixA[j]) * frievaldRandVecrTA[j];
        }
        rTA.push(mySum);
    }
    let rTAPoly = MultilinearPolynomial::<F>::new(rTA.clone());

    let mut rTApt = Vec::new();
    for i in 0..points[0].len()-1{
        rTApt.push(verResCameraHash.1[i]);
    }

    let LHS = rTAPoly.evaluate(&rTApt);
    let mut RHS = imgevals_vec2[0][0] + alpha_1Hash*imgevals_vec2[1][0] + alpha_2Hash*imgevals_vec2[2][0];
    let mut success = true;
    success = success && (verResCameraHash.0 == LHS*RHS);

    for i in 0..4{
        success = success && (hevals_vec2[i][0] == F::ZERO);
        success = success && (prodevals_vec2[i][0] == F::ONE);
    }

    // Implicitely assume that grayscale is really only one channel...
    let mut gray_chan = fieldVec::<F>(&grayImg.R.iter().map(|&x| x as u64).collect::<Vec<_>>());
    let mut padded = gray_chan;
    padded.append(&mut vec![F::ZERO; 1<< numCols]);
    let paddedPoly = MultilinearPolynomial::new(padded);

    let transPt = &points[27];
    let rVal = F::from(30) * imgevals_vec2[0][2];
    let gVal = F::from(59) * imgevals_vec2[1][2];
    let bVal = F::from(11) * imgevals_vec2[2][2];
    let grayVal = F::from(100) * paddedPoly.evaluate(transPt);
    let grayErrVal =  imgevals_vec2[3][1];

    success = success && (grayErrVal == F::from(50) + grayVal -( rVal + gVal + bVal));

    // Verify h and v are done correctly in range check
    
    // Verify the range check
    // Make the embedded tables 
    let primPolyForT = irredPolyTable[numCols] as u64;
    let mut embeddedTable: Vec<F> = vec![F::ZERO; 1 << numCols];
    let mut plusOneTable: Vec<F> = vec![F::ZERO; 1 << numCols];
    //This takes the coefficients of our poly that aren't the most significant one.
    let galoisRep = (primPolyForT) - (1 << numCols);
    //This is how big our table is
    let size = 1 << numCols;
    let mut binaryString: u64 = 1;
    //We create the table by setting index i to g^i(1) where g is our generator.
    for i in 1..(256 as usize + 1) {
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
    let polyTable = MultilinearPolynomial::new(embeddedTable.clone());
    let polyPlusOneTable = MultilinearPolynomial::new(plusOneTable.clone());

    for i in 0..4{
        let myRand = &points[6+i*6];
        let startVal = myRand[0];
        let mut myRandSmall = Vec::new();
        for i in 0..myRand.len()-1{
            myRandSmall.push(myRand[i]);
        }
        let lastVal = myRand[myRand.len()-1];
        let imgAtAlphaSmall = if i < 3 { imgevals_vec2[i][1] } else { imgevals_vec2[i][0] };
        let hAtAlphaRange =  hevals_vec2[i][1];
        let hAtAlphaRangeFiddle = hevals_vec2[i][2];
        let hAtAlphaRange0 =  hevals_vec2[i][3];  
        let prodAtAlphaRange = prodevals_vec2[i][1];
        let fracAtAlphaRange = fracevals_vec2[i][0];
        let prodAtAlphaRange0 = prodevals_vec2[i][2];
        let fracAtAlphaRange0 = fracevals_vec2[i][1];
        let prodAtAlphaRange1 = prodevals_vec2[i][3];
        let fracAtAlphaRange1 = fracevals_vec2[i][2];
        
        // We first compute prod(x) - v(x,0)v(x,1)
        let mut firstHalf = prodAtAlphaRange;

        let myAlpha = myRand[myRand.len()-1]; 
        let vX0 = myAlpha * prodAtAlphaRange0 + (F::ONE- myAlpha) * fracAtAlphaRange0;
        let vX1 =  myAlpha * prodAtAlphaRange1 + (F::ONE- myAlpha) * fracAtAlphaRange1;
        firstHalf += -vX0 * vX1;

        // We are done creating first half

        // alpha0 + merge(I,T)(X) + alpha1 merge(I,T_{+1})(X)

        let mut f1 = alpha1[i] + ((F::ONE- lastVal) * imgAtAlphaSmall + lastVal * (polyTable.evaluate(&myRandSmall)));

        f1 += alpha2[i] * ((F::ONE - lastVal) * imgAtAlphaSmall + lastVal* polyPlusOneTable.evaluate(&myRandSmall));
        // alpha0 + h(X) + alpha1 h_{+1}(X)

        let mut f2 =alpha1[i] + hAtAlphaRange + alpha2[i] * (startVal * hAtAlphaRangeFiddle +  (F::ONE-startVal)* hAtAlphaRange0) ;
        let mut secondHalf = ( f2 * fracAtAlphaRange - f1);
        secondHalf = secondHalf * betas[i];

        let anticipatedVal = verResRangeRGB[i].0;

        let finalVal = firstHalf+secondHalf;

        let extra = eq_eval(&myRand,&maybeChallengeVecs[i]);

        success = success && (anticipatedVal ==finalVal * extra);
    }

    println!("Verifier passed!: {:?}", success);

    println!("verifier done!\n");
}


fn run_full_gray_brake(input_size: usize) {
    // defining various sizes
    let numCols = input_size;
    let numRows = 7;
    let length = numCols + 1;
    
    // setup: get prover and verifier parameters, camera hash
    let (pp, vp, digestRGB) = setup(input_size); 
    
    // now we begin proving
    let prover_start = Instant::now();

    let (hevals_vec, fracevals_vec, prodevals_vec, imgevals_vec, transcript) = prove(pp, input_size, numRows, numCols);

    let elapsed_prover = prover_start.elapsed();
    println!("PROVER TIME: {:?} seconds", elapsed_prover.as_millis() as f64 / 1000 as f64);

    // now verify
    let verifier_start: Instant = Instant::now();

    verify(vp, numRows, numCols, hevals_vec, fracevals_vec, prodevals_vec, imgevals_vec, digestRGB, transcript);

    let elapsed_verifier = verifier_start.elapsed();
    println!("VERIFIER TIME: {:?} seconds", elapsed_verifier.as_millis() as f64 / 1000 as f64);
    
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
        println!("Full System Grayscale, HyperVerITAS Brakedown 256. Size: 2^{:?}\n", i);
        let _res = run_full_gray_brake(i);
        println!("-----------------------------------------------------------------------");
    }
}
