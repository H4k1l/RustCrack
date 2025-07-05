// Copyright 2025 Hakil
// Licensed under the Apache License, Version 2.0

use clap::Parser;
use std::{fs::File, io::{Read, Write}};
use md5;
use sha1::{Sha1, Digest};
use sha2::{Sha256, Sha512};
use cudarc::{self, driver::{CudaContext, LaunchConfig, PushKernelArg}, nvrtc::Ptx};

#[derive(Parser, Debug)]
#[command(about = "RustCrack can crack the hashes of MD5, SHA-1, SHA-256 and SHA-512 or generate simple wordlists, can also use CUDA", long_about = None)]

struct Args {

    // Use the 'clap' library for argument parsing because it is simpler and more ergonomic.

    #[clap(short, long, default_value = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~")]
    chars: String,

    #[clap(short = 'm', long, default_value_t = 1)]
    mnlenght: u64,

    #[clap(short = 'x', long, default_value_t = 10)]
    mxlenght: u64,

    #[clap(short = 'o', long = "output")]
    outputfile: Option<String>,

    #[clap(long = "gen", help = "generate a wordlist")]
    generatewordlist: bool,

    #[clap(short = 'a', long = "algo", help = "the hash algorithm to use (md5/sha-1/sha-256/sha-512)")]
    algorithm: Option<String>,

    #[clap(short = 'w', long, help = "try to use a wordlist to brute the hash")]
    wordlist: Option<String>,

    #[clap(short = 'H', long = "hash")]
    hash: Option<String>,

    #[clap(long = "hashfile")]
    hashfile: Option<String>,

    #[clap(short = 'v', long = "verbose")]
    verbose: bool,

    #[clap(long = "crack", help = "crack a hash")]
    crackhash: bool,

    #[clap(long = "gpu", help = "use CUDA to maximize the efficency for nvidia GPU's")]
    gpu: bool,

    #[clap(short = 't', long, help = "maximum usable of cuda threads in percentage", default_value_t = 100)]
    mxcudathreads: u64,
}
fn main(){
    let args = Args::parse();
    if args.generatewordlist{
        if args.gpu {
            cudageneratewordlist(args.chars.to_owned(), args.mnlenght, args.mxlenght, args.outputfile.clone(), args.mxcudathreads);
        }
        generatewordlist(args.chars.to_owned(), args.mnlenght, args.mxlenght, args.outputfile);
    }
    if args.crackhash{
        if args.gpu {
            cudacrackhash(args.chars, args.mnlenght, args.mxlenght, args.mxcudathreads,args.algorithm.unwrap_or_default(), args.verbose, args.hash.unwrap_or_default(), args.hashfile.unwrap_or_default(), args.wordlist.unwrap_or_default());
        }
        else {
            crackhash(args.chars, args.mnlenght, args.mxlenght, args.algorithm.unwrap_or_default(), args.verbose, args.hash.unwrap_or_default(), args.hashfile.unwrap_or_default(), args.wordlist.unwrap_or_default());
        }
    }
}

fn crackhash(chars: String, mnlenght: u64, mxlenght: u64, mut algo: String, verbose: bool, hash: String, hashfile: String, wordlist: String) {
    if &hash != ""{ // if the hash is provided, try to crack the single hash
        algo = algo.to_lowercase();
        if algo == ""{
            algo = detecthash(&hash);
            println!("rilevated: {}", algo);
        }
        else if algo == "md5" || algo == "sha-1" || algo == "sha-256" || algo == "sha-512" {
            algo = algo;
        }
        else {
            panic!("invalid hash!")
        }
        if wordlist == ""{ // if no wordlist is provided, execute a pure-bruteforce algorithm
            let chars: Vec<char> = chars.chars().collect();
                for length in mnlenght..=mxlenght { // Loop through word lengths from minimum to maximum
                    let total = (chars.len() as u64).pow(length as u32); // Total number of possible combinations for this word length, chars * lenght create a matrix of possibilities
                    for n in 0..total {
                        let mut word = String::new();
                        let mut temp = n;
                        for _ in 0..length { // Generate each character of the word based on current number
                            word.push(chars[(temp % chars.len() as u64) as usize]); // insert in word the expected char(based on the number of position of possibility)
                            temp /= chars.len() as u64; // temp = temp / chars lenght
                        }
                        comparehash(hash.to_owned(), word, algo.to_owned(), verbose);
                }
            }
        }
        else { // else, use the wordlist for a classic wordlist-bruteforce
            let mut filereader = File::open(&wordlist).expect("can't open file");
            let mut file = String::new();
            filereader.read_to_string(&mut file).expect("can't read file");
            for word in file.lines(){
                comparehash(hash.to_owned(), word.to_string(), algo.to_owned(), verbose);
            }
        }
    }
    else { // else, get the hashes file, and crack all of them
        let chars: Vec<char> = chars.chars().collect();
        let mut filereader = File::open(hashfile).expect("can't open file");
        let mut file = String::new();
        filereader.read_to_string(&mut file).expect("can't read file");
        for hash in file.lines(){
            println!("trying '{}'", hash);
            algo = detecthash(&hash);
            println!("rilevated: {}", algo);
            if &wordlist == ""{ // if no wordlist is provided, execute a pure-bruteforce algorithm
                let mut found = false; 
                for length in mnlenght..=mxlenght { // Loop through word lengths from minimum to maximum
                    let total = (chars.len() as u64).pow(length as u32); // Total number of possible combinations for this word length, chars * lenght create a matrix of possibilities
                    for n in 0..total {
                        let mut word = String::new();
                        let mut temp = n;
                        for _ in 0..length { // Generate each character of the word based on current number
                            word.push(chars[(temp % chars.len() as u64) as usize]); // insert in word the expected char(based on the number of position of possibility)
                            temp /= chars.len() as u64; // temp = temp / chars lenght
                        }
                        if comparehash(hash.to_owned(), word, algo.to_owned(), verbose){
                            found = true;
                            break;
                        }
                    }
                    if found{
                        break;
                    }
                }
            }
            else{ // else, use the wordlist for a classic wordlist-bruteforce
                let mut filereader = File::open(&wordlist).expect("can't open file");
                let mut file = String::new();
                filereader.read_to_string(&mut file).expect("can't read file");
                for word in file.lines(){
                    if comparehash(hash.to_string(), word.to_string(), algo.to_owned(), verbose){
                        break;
                    }
                }
            }
        }   
    }
}

fn cudacrackhash(chars: String, mnlenght: u64, mxlenght: u64, mxcudathreads: u64, mut algo: String, verbose: bool, hash: String, hashfile: String, wordlist: String) {
    
    let mut hashVecRAM: Vec<&str> = vec![&hash];
    let mut lines = String::new();

    if &hash == "" {
        let mut filereader = File::open(hashfile).expect("can't open file");
        filereader.read_to_string(&mut lines).expect("can't read file");
        hashVecRAM = lines.lines().collect(); // the hash in a vector
    }
    
    algo = algo.to_lowercase();
    if algo == ""{
        if &hash != "" {
            algo = detecthash(&hash);
        }
        else {
            algo = detecthash(hashVecRAM[0]);
        }
        println!("rilevated: {}", algo);
    }
    else if algo == "md5" || algo == "sha-1" || algo == "sha-256" || algo == "sha-512" {
        algo = algo;
    }
    else {
        panic!("invalid hash!")
    }

    let mut hashtype: i32 = 0;

    if algo == "md5" {
        hashtype = 0;
    }
    else if algo == "sha-1" {
        hashtype = 1;
    }
    else if algo == "sha-256" {
        hashtype = 2;
    }
    else if algo == "sha-512" {
        hashtype = 3;
    }

    if wordlist == ""{ // if no wordlist is provided, execute a pure-bruteforce algorithm

        let ctx = CudaContext::new(0).expect("can't create cuda context");

        // calculate the maximum threads for chunking
        let mut mxThread: u64 = 0;
        unsafe {
            let totSM = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap(); 
            let smMxThread = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR).unwrap(); 
            mxThread = (totSM * smMxThread) as u64;
        }
        mxThread = (mxThread * mxcudathreads) / 100; // usable amount of threads (as a percentage)
        
        let mut rangeVecRAM: Vec<u64> = Vec::new();
        let chars: Vec<char> = chars.chars().collect();
        let chars: String = String::from_iter(chars);

        // calculation of total actions to be performed by the GPU
        let mut totalAct: u64 = 0; 
        for length in mnlenght..=mxlenght {
            for _ in 0..=((chars.len() as u64).pow(length as u32)) {
                totalAct += 1;
            }
        }
        let actForThread = totalAct.div_ceil(mxThread);

        // structure of rangeVecRAM: <length, from, to>. Each thread works on a group of these 3 elements
        for length in mnlenght..=mxlenght {
            for i in 0..((chars.len() as u64).pow(length as u32)) {
                if (i * actForThread) > totalAct || !((i * actForThread) + (actForThread - 1) <= (chars.len() as u64).pow(length as u32)){
                    break;
                }
                rangeVecRAM.push(length);
                rangeVecRAM.push(i * actForThread);
                if (i * actForThread) + (actForThread - 1) <= (chars.len() as u64).pow(length as u32) {
                    rangeVecRAM.push((i * actForThread) + (actForThread - 1));
                }
                else {
                    break;
                }
            }
        }

        // starting the kernel with cuda
        let ptx = ctx.load_module(Ptx::from_file("src/gpuKernel/PureBrute.ptx")).expect("error while loading the module, be sure the module 'src/gpuKernel/WordlistBrute.ptx' exist");
        let stream = ctx.default_stream();
        let fnct = ptx.load_function("crackWord").unwrap();

        // converting in bytes
        let hashVecRAMBytes = hashVecRAM.join("").as_bytes().to_vec();
        let charsVecRAMBytes = chars.as_bytes().to_vec();
        let n = rangeVecRAM.len();
        let mut usedThreads = rangeVecRAM.len() / 3;
        let chrlen = chars.len();
        
        // loading the memory
        let mut hashListDRAM = stream.alloc_zeros::<u8>(hashVecRAMBytes.len()).expect("can't allocate the hash list in the DRAM");
        let mut charsListDRAM = stream.alloc_zeros::<u8>(charsVecRAMBytes.len()).expect("can't allocate the chars list in the DRAM");
        let mut rangesDRAM = stream.alloc_zeros::<u64>(rangeVecRAM.len()).expect("can't allocate the ranges list in the DRAM");

        // allocate the memory in the gpu
        stream.memcpy_htod(&hashVecRAMBytes, &mut hashListDRAM).expect("can't allocate the wordlist in the DRAM");
        stream.memcpy_htod(&charsVecRAMBytes, &mut charsListDRAM).expect("can't allocate the wordlist in the DRAM");
        stream.memcpy_htod(&rangeVecRAM, &mut rangesDRAM).expect("can't allocate the wordlist in the DRAM");
        stream.synchronize();

        // adapting the launcher to the maximum number of threads per block
        let mut mxthreadperblock = 0;
        unsafe {
            mxthreadperblock = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
        }
        
        let mut usedBlocks = 1;
        if usedThreads > mxthreadperblock as usize{
            // if the threads are more than the maximum allowed, split the threads in blocks, checking the occupancy
            mxthreadperblock = calulateoptimalblocksize();
            usedBlocks = usedThreads.div_ceil(mxthreadperblock as usize);
            usedThreads = mxthreadperblock as usize;
        }
        
        let launcher = LaunchConfig {
            grid_dim: (usedBlocks as u32, 1, 1),
            block_dim: (usedThreads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // launch the kernel
        let mut launchArg = stream.launch_builder(&fnct);
        launchArg.arg(&mut charsListDRAM);
        launchArg.arg(&chrlen);
        launchArg.arg(&mut rangesDRAM);
        launchArg.arg(&mut hashListDRAM);
        launchArg.arg(&hashtype);
        launchArg.arg(&n);
        launchArg.arg(&verbose);
        unsafe {launchArg.launch(launcher);}

        stream.synchronize().expect("can't synchronize the threads gpu");

    }   
    else { // else, use the wordlist for a classic wordlist-bruteforce
        let mut filereader = File::open(&wordlist).expect("can't open file");
        let mut file = String::new();
        filereader.read_to_string(&mut file).expect("can't read file");

        // starting the kernel with cuda
        let ctx = CudaContext::new(0).expect("can't create cuda context");
        let ptx = ctx.load_module(Ptx::from_file("src/gpuKernel/WordlistBrute.ptx")).expect("error while loading the module, be sure the module 'src/gpuKernel/WordlistBrute.ptx' exist");
        let stream = ctx.default_stream();
        let fnct = ptx.load_function("crackWord").unwrap();

        // loading the memory
        let wordListVecRAM: Vec<&str> = file.lines().collect(); // all the wordlist in a vector
        let mut x: usize = wordListVecRAM.len();

        let mut offsetVecRAM: Vec<i32> = vec![0]; // all the offsets of the word
        let mut offsetCalc: usize = 0;
        for word in &wordListVecRAM {
            offsetCalc += word.len() ;
            offsetCalc += 1;
            offsetVecRAM.push(offsetCalc as i32);
        }

        let lenghtsVecRAM: Vec<i32> = wordListVecRAM.iter().map(|x| x.len() as i32).collect(); // all the lenghts

        let y: usize = hashVecRAM.len();
        let n = x * y;


        // converting in bytes
        let hashVecRAMBytes = hashVecRAM.join("").as_bytes().to_vec();
        let byteswordListVecRAM = wordListVecRAM.join("\0").as_bytes().to_vec();
        
        // allocate the memory in the gpu
        let mut wordListVecDRAM = stream.alloc_zeros::<u8>(byteswordListVecRAM.len()).expect("can't allocate the wordlist in the DRAM");
        let mut offsetVecDRAM = stream.alloc_zeros::<i32>(offsetVecRAM.len()).expect("can't allocate the offset list in the DRAM");
        let mut lenghtsVecDRAM = stream.alloc_zeros::<i32>(lenghtsVecRAM.len()) .expect("can't allocate the lenghts list in the DRAM");
        let mut hashListDRAM = stream.alloc_zeros::<u8>(hashVecRAMBytes.len()).expect("can't allocate the hash list in the DRAM");

        stream.memcpy_htod(&byteswordListVecRAM, &mut wordListVecDRAM).expect("can't allocate the wordlist in the DRAM");
        stream.memcpy_htod(&offsetVecRAM, &mut offsetVecDRAM).expect("can't allocate the wordlist in the DRAM");
        stream.memcpy_htod(&lenghtsVecRAM, &mut lenghtsVecDRAM).expect("can't allocate the wordlist in the DRAM");
        stream.memcpy_htod(&hashVecRAMBytes, &mut hashListDRAM).expect("can't allocate the wordlist in the DRAM");

        // adapting the launcher to the maximum number of threads per block
        let mut mxthreadperblock = 0;
        unsafe {
            mxthreadperblock = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
        }
        
        let mut usedBlocks = 1;
        if x > mxthreadperblock as usize{
            // if the threads are more than the maximum allowed, split the threads in blocks, checking the occupancy
            mxthreadperblock = calulateoptimalblocksize();
            usedBlocks = x.div_ceil(mxthreadperblock as usize);
            x = mxthreadperblock as usize;
        }
        
        // launch the kernel
        let launcher = LaunchConfig {
            grid_dim: (usedBlocks as u32, 1, 1),
            block_dim: (x as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut launchArg = stream.launch_builder(&fnct);
        launchArg.arg(&mut wordListVecDRAM);
        launchArg.arg(&mut offsetVecDRAM);
        launchArg.arg(&mut lenghtsVecDRAM);
        launchArg.arg(&mut hashListDRAM);
        launchArg.arg(&hashtype);
        launchArg.arg(&n);
        launchArg.arg(&verbose);
        unsafe {launchArg.launch(launcher);}

        stream.synchronize().expect("can't synchronize the threads gpu");
    }
}

fn generatewordlist(chars: String, mnlenght: u64, mxlenght: u64, outputfile: Option<String>) { // same algorithm as before for the pure-bruteforcing
    let mut filewriter: Option<File> = None;
    if let Some(ref output) = outputfile {
        filewriter = Some(File::create(output).expect("cant create file"));
    }
    let chars: Vec<char> = chars.chars().collect();
    
    for length in mnlenght..=mxlenght { // Loop through word lengths from minimum to maximum
        let total = (chars.len() as u64).pow(length as u32); // Total number of possible combinations for this word length, chars * lenght create a matrix of possibilities
        for n in 0..total {
            let mut word = String::new();
            let mut temp = n;
            for _ in 0..length { // Generate each character of the word based on current number
                word.push(chars[(temp % chars.len() as u64) as usize]); // insert in word the expected char(based on the number of position of possibility)
                temp /= chars.len() as u64; // temp = temp / chars lenght
            }
            if let Some(ref mut writer) = filewriter {
                word.push_str("\n");
                writer.write_all(word.as_bytes()).expect("Failed to write to file");
            } else {
                // println!("{}", word);
            }

        }

    }

}

fn cudageneratewordlist(chars: String, mnlenght: u64, mxlenght: u64, outputfile: Option<String>, mxcudathreads: u64) {
    let ctx = CudaContext::new(0).expect("can't create cuda context");

    // calculate the maximum threads for chunking
    let mut mxThread: u64 = 0;
    unsafe {
        let totSM = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap(); 
        let smMxThread = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR).unwrap(); 
        mxThread = (totSM * smMxThread) as u64;
    }
    mxThread = (mxThread * mxcudathreads) / 100; // usable amount of threads (as a percentage)
    
    let mut rangeVecRAM: Vec<u64> = Vec::new();
    let chars: Vec<char> = chars.chars().collect();
    let chars: String = String::from_iter(chars);

    // calculation of total actions to be performed by the GPU
    let mut totalAct: u64 = 0; 
    let mut ttact: u64 = 0;
    for length in mnlenght..=mxlenght {
        for _ in 0..=((chars.len() as u64).pow(length as u32)) {
            totalAct += 1;
            ttact += 1;
        }
        ttact += length*(chars.len() as u64).pow(length as u32);
    }
    let actForThread = totalAct.div_ceil(mxThread);

    // structure of rangeVecRAM: <length, from, to>. Each thread works on a group of these 3 elements
    for length in mnlenght..=mxlenght {
        for i in 0..((chars.len() as u64).pow(length as u32)) {
            if (i * actForThread) > totalAct || !((i * actForThread) + (actForThread - 1) <= (chars.len() as u64).pow(length as u32)){
                break;
            }
            rangeVecRAM.push(length);
            rangeVecRAM.push(i * actForThread);
            if (i * actForThread) + (actForThread - 1) <= (chars.len() as u64).pow(length as u32) {
                rangeVecRAM.push((i * actForThread) + (actForThread - 1));
            }
            else {
                break;
            }
        }
    }

    // starting the kernel with cuda
    let ptx = ctx.load_module(Ptx::from_file("src/gpuKernel/WordlistGen.ptx")).expect("error while loading the module, be sure the module 'src/gpuKernel/WordlistBrute.ptx' exist");
    let stream = ctx.default_stream();
    let fnct = ptx.load_function("genWord").unwrap();

    // converting in bytes
    let charsVecRAMBytes = chars.as_bytes().to_vec();
    let n = rangeVecRAM.len();
    let mut usedThreads = rangeVecRAM.len() / 3;
    let chrlen = chars.len();

    // loading the memory
    let mut charsListDRAM = stream.alloc_zeros::<u8>(charsVecRAMBytes.len()).expect("can't allocate the chars list in the DRAM");
    let mut rangesDRAM = stream.alloc_zeros::<u64>(rangeVecRAM.len()).expect("can't allocate the ranges list in the DRAM");
    let mut outputDRAM = stream.alloc_zeros::<u8>(ttact as usize).expect("can't allocate the output vec in the DRAM");

    // allocate the memory in the gpu
    stream.memcpy_htod(&charsVecRAMBytes, &mut charsListDRAM).expect("can't allocate the wordlist in the DRAM");
    stream.memcpy_htod(&rangeVecRAM, &mut rangesDRAM).expect("can't allocate the wordlist in the DRAM");
    stream.synchronize();

    // adapting the launcher to the maximum number of threads per block
    let mut mxthreadperblock = 0;
    unsafe {
        mxthreadperblock = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
    }
    
    let mut usedBlocks = 1;
    if usedThreads > mxthreadperblock as usize{
        // if the threads are more than the maximum allowed, split the threads in blocks, checking the occupancy
        mxthreadperblock = calulateoptimalblocksize();
        usedBlocks = usedThreads.div_ceil(mxthreadperblock as usize);
        usedThreads = mxthreadperblock as usize;
    }
    
    let launcher = LaunchConfig {
        grid_dim: (usedBlocks as u32, 1, 1),
        block_dim: (usedThreads as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // launch the kernel
    let mut launchArg = stream.launch_builder(&fnct);
    launchArg.arg(&mut charsListDRAM);
    launchArg.arg(&chrlen);
    launchArg.arg(&mut rangesDRAM);
    launchArg.arg(&n);
    launchArg.arg(&outputDRAM);
    unsafe {launchArg.launch(launcher);}

    stream.synchronize().expect("can't synchronize the threads gpu");

    let outputVecRAM = stream.memcpy_dtov(&outputDRAM).unwrap();
    let otpstr = String::from_utf8(outputVecRAM).unwrap();

    let mut filewriter: Option<File> = None;
    if let Some(ref output) = outputfile {
        filewriter = Some(File::create(output).expect("cant create file"));

    }
    

}

fn calulateoptimalblocksize() -> i32{
    let mut mxthreadperblock = 0; 
    let mut mxblockspermulti = 0;
    unsafe {
        mxthreadperblock = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
        mxblockspermulti = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR).unwrap();
    }

    let mut bestoccupancy = 0.0;
    let mut bestblocksize = 0;

    for i in (32..=mxthreadperblock).step_by(32) {
        let wrpperblock = i / 32;
        let blockpersm = std::cmp::min(48  / wrpperblock, mxblockspermulti);
        let actwrp = blockpersm * wrpperblock;
        let occupan = (actwrp as f32) / (48 as f32);
        if occupan > bestoccupancy {
            bestoccupancy = occupan; 
            bestblocksize = i;
        }
    }
    bestblocksize
}

fn detecthash(hash: &str) -> String{
    if hash.chars().all(|c| c.is_digit(16)){
        if hash.len() == 32 {
            return "md5".to_string();
        }
        else if hash.len() == 40{
            return "sha-1".to_string();
        }
        else if hash.len() == 64{
            return "sha-256".to_string();
        }
        else if hash.len() == 128{
            return "sha-512".to_string();
        }
        else {
            panic!("invalid hash!");
        }
    }
    else{
        panic!("invalid hash!");
    }
}

fn comparehash(hash: String, word: String, algo: String, verbose: bool) -> bool{
    if algo == "md5" && format!("{:x}",md5::compute(&word)) == hash{
        println!("FOUND MATCH: '{}' -> '{}'", hash, word);
        return true;
    }
    else if algo == "sha-1"{
        let mut sha1_hasher = Sha1::new();
        sha1_hasher.update(&word);
        let sha_result = sha1_hasher.finalize();
        if format!("{:x}", sha_result) == hash{
            println!("FOUND MATCH: '{}' -> '{}'", hash, word);
            return true;
        }
    }
    else if algo == "sha-256"{
        let mut sha256_hasher = Sha256::new();
        sha256_hasher.update(&word);
        let sha_result = sha256_hasher.finalize();
        if format!("{:x}", sha_result) == hash{
            println!("FOUND MATCH: '{}' -> '{}'", hash, word);
            return true;
        }
    }
    else if algo == "sha-512"{
        let mut sha512_hasher = Sha512::new();
        sha512_hasher.update(&word);
        let sha_result = sha512_hasher.finalize();
        if format!("{:x}", sha_result) == hash{
            println!("FOUND MATCH: '{}' -> '{}'", hash, word);
            return true;
        }
    }
    if verbose{
        println!("dont work: '{}'", word);
    }
    return false;

}
