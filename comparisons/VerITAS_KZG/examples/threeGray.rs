use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::circuit_data::CircuitData;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::ProofWithPublicInputs;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::env;


// static PIXELS : usize = 1<<14;

fn print_time_since(last: u128, tag: &str) -> u128 {
    let now = SystemTime::now();
    let now_epoc = now
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let now = now_epoc.as_millis();
    println!("{:?} - time since last check: {:?}", tag, (now - last) as f32 / 60000.0); 
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


pub fn grayscale(size: &usize) -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let mut r_vals = Vec::new();
    let mut g_vals = Vec::new();
    let mut b_vals = Vec::new();
    let mut x_vals = Vec::new();
    let mut rem_vals = Vec::new();
    let file = read_photo("./images/Veri", size, "R");
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<u32>().unwrap();
            r_vals.push(i as u32);
    }
    let file = read_photo("./images/Veri", size, "G");
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<u32>().unwrap();
            g_vals.push(i as u32);
    }
    let file = read_photo("./images/Veri", size, "B");
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<u32>().unwrap();
            b_vals.push(i as u32);
    }
    for i in 0..1<<size {
        
        let r_f = r_vals[i] as f64;
        let g_f = g_vals[i] as f64;
        let b_f = b_vals[i] as f64;

        let x_f = 0.3 * r_f + 0.59 * g_f + 0.11 * b_f;
        x_vals.push(x_f.round() as i32);

        rem_vals.push((r_vals[i] * 30 + g_vals[i] * 59 + b_vals[i] * 11) as i32 - 100 * x_vals[i]);
    }
   
     // Timing setup
    let start = SystemTime::now();
    let start_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let start = start_epoch.as_millis();
    let mut last = start;

    let mut config = CircuitConfig::standard_recursion_config();
    config.zero_knowledge = true;
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let mut pw = PartialWitness::new();

    let mut r_targets = Vec::new();
    let mut g_targets = Vec::new();
    let mut b_targets = Vec::new();
    
   
    for _ in 0..1<<size {
        let r = builder.add_virtual_target();
        r_targets.push(r);

        let g = builder.add_virtual_target();
        g_targets.push(g);

        let b = builder.add_virtual_target();
        b_targets.push(b);

        let mut all = Vec::new();

        all.push(builder.mul_const(F::from_canonical_u32(30), r));
        all.push(builder.mul_const(F::from_canonical_u32(59), g));
        all.push(builder.mul_const(F::from_canonical_u32(11), b));

        let s = builder.add_many(all);
        builder.register_public_input(s);
    }

    let data = builder.build::<C>();
    last = print_time_since(last, "setup done"); 

    for i in 0..1<<size {
        pw.set_target(r_targets[i], F::from_canonical_u32(r_vals[i]));
        pw.set_target(g_targets[i], F::from_canonical_u32(g_vals[i]));
        pw.set_target(b_targets[i], F::from_canonical_u32(b_vals[i]));
    }

    let proof = data.prove(pw)?;
    last = print_time_since(last, "proof done"); 

    for i in 0..1<<size {
        assert!((proof.public_inputs[i].0) as i32 == 100 * x_vals[i] + rem_vals[i])
    }

    let res = data.verify(proof);
    let _ = res.unwrap();

    _ = print_time_since(last, "verify done"); 

    Ok(())
}

pub fn grayscale_system_setup(size: &usize) -> (Vec<i32>, Vec<i32>, Vec<Target>, Vec<Target>, Vec<Target>, Vec<u32>, Vec<u32>, Vec<u32>, CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>, PartialWitness<GoldilocksField>) {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let mut r_vals = Vec::new();
    let mut g_vals = Vec::new();
    let mut b_vals = Vec::new();
    let mut x_vals = Vec::new();
    let mut rem_vals = Vec::new();
    let file = read_photo("./images/Veri", size, "R");
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<u32>().unwrap();
            r_vals.push(i as u32);
    }
    let file = read_photo("./images/Veri", size, "G");
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<u32>().unwrap();
            g_vals.push(i as u32);
    }
    let file = read_photo("./images/Veri", size, "B");
    for line in file.lines() {
        let line = line.expect("Unable to read line");
        let i = line.parse::<u32>().unwrap();
            b_vals.push(i as u32);
    }
    for i in 0..1<<size {
        
        let r_f = r_vals[i] as f64;
        let g_f = g_vals[i] as f64;
        let b_f = b_vals[i] as f64;

        let x_f = 0.3 * r_f + 0.59 * g_f + 0.11 * b_f;
        x_vals.push(x_f.round() as i32);

        rem_vals.push((r_vals[i] * 30 + g_vals[i] * 59 + b_vals[i] * 11) as i32 - 100 * x_vals[i]);
    }
   
     // Timing setup
    let start = SystemTime::now();
    let start_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let start = start_epoch.as_millis();
    let mut last = start;

    let mut config = CircuitConfig::standard_recursion_config();
    config.zero_knowledge = true;
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let mut pw = PartialWitness::new();

    let mut r_targets = Vec::new();
    let mut g_targets = Vec::new();
    let mut b_targets = Vec::new();
    
   
    for _ in 0..1<<size {
        let r = builder.add_virtual_target();
        r_targets.push(r);

        let g = builder.add_virtual_target();
        g_targets.push(g);

        let b = builder.add_virtual_target();
        b_targets.push(b);

        let mut all = Vec::new();

        all.push(builder.mul_const(F::from_canonical_u32(30), r));
        all.push(builder.mul_const(F::from_canonical_u32(59), g));
        all.push(builder.mul_const(F::from_canonical_u32(11), b));

        let s = builder.add_many(all);
        builder.register_public_input(s);
    }

    let data = builder.build::<C>();
    last = print_time_since(last, "setup done"); 

    return (x_vals, rem_vals, r_targets, g_targets, b_targets, r_vals, g_vals, b_vals, data, pw)
}

pub fn grayscale_system_prove(size: &usize, 
    r_targets: Vec<Target>, g_targets: Vec<Target>, b_targets: Vec<Target>, 
    r_vals: Vec<u32>, g_vals: Vec<u32>, b_vals: Vec<u32>,
    data: CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>, mut pw: PartialWitness<GoldilocksField>
    ) -> (CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>, ProofWithPublicInputs<GoldilocksField, PoseidonGoldilocksConfig, 2>)
    
    {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    
    for i in 0..1<<size {
        pw.set_target(r_targets[i], F::from_canonical_u32(r_vals[i]));
        pw.set_target(g_targets[i], F::from_canonical_u32(g_vals[i]));
        pw.set_target(b_targets[i], F::from_canonical_u32(b_vals[i]));
    }

    let proof = data.prove(pw).unwrap();

    return (data, proof)
}  

pub fn grayscale_system_verify(data: CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>, proof: ProofWithPublicInputs<GoldilocksField, PoseidonGoldilocksConfig, 2>, x_vals: Vec<i32>, rem_vals: Vec<i32>, size: &usize) {

    for i in 0..1<<size {
        assert!((proof.public_inputs[i].0) as i32 == 100 * x_vals[i] + rem_vals[i])
    }

    let res = data.verify(proof);
    let _ = res.unwrap();
}

fn main(){
    let args: Vec<String> = env::args().collect();

    let first_size = args[1].parse::<usize>().unwrap();
    let mut last_size = first_size;
    if args.len() == 3{
        last_size = args[2].parse::<usize>().unwrap();
    }

    for i in first_size..last_size+1 {
        println!("Three Channel Grayscale, VerITAS KZG. Size: 2^{:?}\n", i);
        let now = Instant::now();
        let _res = grayscale(&i);
        let elapsed_time = now.elapsed();
        println!("Whole Time: {:?} seconds", elapsed_time.as_millis() as f64 / 1000 as f64);
        println!("-----------------------------------------------------------------------");
    }

}

