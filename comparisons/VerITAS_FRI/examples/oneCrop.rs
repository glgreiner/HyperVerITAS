use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::CompressedProofWithPublicInputs;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::env;


fn print_time_since(last: u128, tag: &str) -> u128 {
    let now = SystemTime::now();
    let now_epoc = now
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let now = now_epoc.as_millis();
    println!("{:?} - time since last check: {:?} seconds", tag, (now - last) as f64 / 1000 as f64); 
    return now;
} 

fn get_filename(prefix: &str, size: &usize, postfix: &str) -> String {
    let mut filename = prefix.to_owned();
    filename.push_str(&size.to_string());
    filename.push_str(postfix);
    filename.push_str(".txt");
    return filename
}

fn read_photo(prefix: &str, size: &usize, postfix: &str) -> BufReader<File> {
    let file = File::open(get_filename(prefix, &size, postfix)).expect("Unable to open file");
    return BufReader::new(file);
}

pub fn crop_system_setup(size: &usize, postfix:&str) 
-> (Vec<u32>, Vec<u32>, Vec<Target>, CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>, PartialWitness<GoldilocksField>) {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    // Timing setup
    let start = SystemTime::now();
    let start_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let start = start_epoch.as_millis();
    let mut last = start;

    let mut w_r_vals = Vec::new();
    let mut x_r_vals = Vec::new();
    
    let file = read_photo("./images/Veri", size, postfix);
        for line in file.lines() {
            let line = line.expect("Unable to read line");
            let i = line.parse::<u32>().unwrap();
                w_r_vals.push(i as u32);
        }

    println!("{}",w_r_vals.len());
    for i in 0..1<<(size-1) {
        x_r_vals.push(w_r_vals[i]);
    }

    last = print_time_since(last, "values generated"); 
   
    let mut config = CircuitConfig::standard_recursion_config();
    config.zero_knowledge = true;
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let mut pw: PartialWitness<GoldilocksField> = PartialWitness::new();

    let mut w_r_targets= Vec::new();

    for _ in 0..1<<(size-1) {
        let r = builder.add_virtual_target();
        w_r_targets.push(r);
        builder.register_public_input(r);    
    }
        
    let data = builder.build::<C>();
    last = print_time_since(last, "setup done"); 

    return (x_r_vals, w_r_vals, w_r_targets, data, pw)
}

pub fn crop_system_prove(size: &usize, w_r_vals: Vec<u32>, w_r_targets: Vec<Target>, data: CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>, mut pw: PartialWitness<GoldilocksField>) 
-> (CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>, CompressedProofWithPublicInputs<GoldilocksField, PoseidonGoldilocksConfig, 2>) {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    // Timing setup
    let start = SystemTime::now();
    let start_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let start = start_epoch.as_millis();
    let mut last = start;
    
    for i in 0..1<<(size-1) {
        pw.set_target(w_r_targets[i], F::from_canonical_u32(w_r_vals[i]));
    }

    let proof = data.prove(pw).unwrap();
    let compressed_proof = data.compress(proof).unwrap();

    last = print_time_since(last, "proof done");

    (data, compressed_proof)
}

pub fn crop_system_verify(data: CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>, compressed_proof: CompressedProofWithPublicInputs<GoldilocksField, PoseidonGoldilocksConfig, 2>, x_r_vals: Vec<u32>) {
    let decompressed_compressed_proof = data.decompress(compressed_proof).unwrap();

    for i in 0..decompressed_compressed_proof.public_inputs.len() {
        assert!((decompressed_compressed_proof.public_inputs[i].0) as u32 == x_r_vals[i]);
    }

    let res = data.verify(decompressed_compressed_proof);
    let _ = res.unwrap();
}

pub fn crop_system(size: &usize, postfix:&str) -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    // Timing setup
    let start = SystemTime::now();
    let start_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let start = start_epoch.as_millis();
    let mut last = start;

    let mut w_r_vals = Vec::new();
    let mut x_r_vals = Vec::new();
    
    let file = read_photo("./images/Veri", size, postfix);
        for line in file.lines() {
            let line = line.expect("Unable to read line");
            let i = line.parse::<u32>().unwrap();
                w_r_vals.push(i as u32);
        }

    println!("{}",w_r_vals.len());
    for i in 0..1<<(size-1) {
        x_r_vals.push(w_r_vals[i]);
    }

    last = print_time_since(last, "values generated"); 
   
    let mut config = CircuitConfig::standard_recursion_config();
    config.zero_knowledge = true;
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let mut pw = PartialWitness::new();

    let mut w_r_targets = Vec::new();

    for _ in 0..1<<(size-1) {
        let r = builder.add_virtual_target();
        w_r_targets.push(r);
        builder.register_public_input(r);    
    }
        

    let data = builder.build::<C>();
    last = print_time_since(last, "setup done"); 

    for i in 0..1<<(size-1) {
        pw.set_target(w_r_targets[i], F::from_canonical_u32(w_r_vals[i]));
    }

    let proof = data.prove(pw)?;
    let compressed_proof = data.compress(proof)?;

    last = print_time_since(last, "proof done");

    let decompressed_compressed_proof = data.decompress(compressed_proof)?;


    for i in 0..decompressed_compressed_proof.public_inputs.len() {
        assert!((decompressed_compressed_proof.public_inputs[i].0) as u32 == x_r_vals[i]);
    }

    let res = data.verify(decompressed_compressed_proof);
    let _ = res.unwrap();

    
    _ = print_time_since(last, "verify done"); 

    Ok(())
}

fn main(){
    let args: Vec<String> = env::args().collect();

    let first_size = args[1].parse::<usize>().unwrap();
    let mut last_size = first_size;
    if args.len() == 3{
        last_size = args[2].parse::<usize>().unwrap();
    }

    for i in first_size..last_size+1 {
        println!("One Channel Crop, VerITAS FRI. Size: 2^{:?}\n", i);
        let now = Instant::now();
        let _res = crop_system(&i, "R");
        let elapsed_time = now.elapsed();
        println!("Whole Time: {:?} seconds", elapsed_time.as_millis() as f64 / 1000 as f64);
        println!("-----------------------------------------------------------------------");
    }
}