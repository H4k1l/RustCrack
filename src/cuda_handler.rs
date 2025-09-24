// Copyright 2025 Hakil
// Licensed under the Apache License, Version 2.0

use crate::generators::range_builder;
// importing modules
use crate::hash_utils;
use crate::generators;

use std::vec;
// importing libraries
use std::{
    fs::File, 
    io::{
        Read, 
        Write
    }, 
};
use cudarc::{
    self,
    driver::{
        CudaContext, 
        LaunchConfig, 
        PushKernelArg
    }, 
    nvrtc::Ptx
};

fn calulate_optimal_block_size() -> i32 {
    let mxthread_per_block: i32; 
    let mxblocks_per_multi: i32;
    unsafe {
        mxthread_per_block = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
        mxblocks_per_multi = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR).unwrap();
    }

    let mut bestoccupancy = 0.0;
    let mut bestblocksize = 0;

    for i in (32..=mxthread_per_block).step_by(32) {
        let wrpperblock = i / 32;
        let blockpersm = std::cmp::min(48  / wrpperblock, mxblocks_per_multi);
        let actwrp = blockpersm * wrpperblock;
        let occupan = (actwrp as f32) / (48 as f32);
        if occupan > bestoccupancy {
            bestoccupancy = occupan; 
            bestblocksize = i;
        }
    }
    bestblocksize
}

pub fn cuda_generate_wordlist(chars: String, mnlength: u64, mxlength: u64, outputfile: Option<String>, mxcudathreads: u64, verbose: u8) {
    let ctx = CudaContext::new(0).expect("can't create cuda context");

    // calculate the maximum threads for chunking
    let mut mxthread: u64;
    unsafe {
        let tot_sm = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap(); 
        let sm_mxthread = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR).unwrap(); 
        mxthread = (tot_sm * sm_mxthread) as u64;
    }
    mxthread = (mxthread * mxcudathreads) / 100; // usable amount of threads (as a percentage)
    
    let mut range_vec_ram: Vec<u64> = Vec::new();
    let chars: Vec<char> = chars.chars().collect();
    let chars: String = String::from_iter(chars);

    // calculation of total actions to be performed by the GPU
    let mut total_act: u64 = 0; 
    let mut ttact: u64 = 0;
    for length in mnlength..=mxlength {
        for _ in 0..=((chars.len() as u64).pow(length as u32)) {
            total_act += 1;
            ttact += 1;
        }
        ttact += length*(chars.len() as u64).pow(length as u32);
    }
    let act_for_thread = total_act.div_ceil(mxthread);

    // structure of range_vec_ram: <length, from, to>. Each thread works on a group of these 3 elements
    for length in mnlength..=mxlength {
        for i in 0..((chars.len() as u64).pow(length as u32)) {
            if (i * act_for_thread) > total_act || !((i * act_for_thread) + (act_for_thread - 1) <= (chars.len() as u64).pow(length as u32)){
                break;
            }
            range_vec_ram.push(length);
            range_vec_ram.push(i * act_for_thread);
            if (i * act_for_thread) + (act_for_thread - 1) <= (chars.len() as u64).pow(length as u32) {
                range_vec_ram.push((i * act_for_thread) + (act_for_thread - 1));
            }
            else {
                break;
            }
        }
    }

    // starting the kernel with cuda
    let ptx = ctx.load_module(Ptx::from_file("src/gpuKernel/wordlist_gen.ptx")).expect("error while loading the module, be sure the module 'src/gpuKernel/WordlistBrute.ptx' exist");
    let stream = ctx.default_stream();
    let fnct = ptx.load_function("gen_word").unwrap();

    // converting in bytes
    let chars_vec_ram_bytes = chars.as_bytes().to_vec();
    let mut used_threads = range_vec_ram.len() / 3;
    let chrlen = chars.len();

    if verbose >= 1{
        let mut total_used_mem = 0;
        total_used_mem += std::mem::size_of::<Vec<u64>>() + ttact as usize * std::mem::size_of::<u64>();
        total_used_mem += std::mem::size_of::<Vec<u8>>() + chars_vec_ram_bytes.capacity() * std::mem::size_of::<u8>();
        total_used_mem += std::mem::size_of::<Vec<u64>>() + range_vec_ram.capacity() * std::mem::size_of::<u64>();
        let (meminfo1, _) = cudarc::driver::result::mem_get_info().unwrap();
        println!("memory required: {total_used_mem} bytes\nusable memory of the gpu: {meminfo1} bytes\ntotal actions to perform: {total_act}\ntotal actions to perform for each thread: {act_for_thread}");
    }

    // loading the memory
    let mut char_list_dram = stream.alloc_zeros::<u8>(chars_vec_ram_bytes.len()).expect("can't allocate the chars list in the DRAM");
    let mut ranges_dram = stream.alloc_zeros::<u64>(range_vec_ram.len()).expect("can't allocate the ranges list in the DRAM");
    let output_dram = stream.alloc_zeros::<u8>(ttact as usize).expect("can't allocate the output vec in the DRAM");

    // allocate the memory in the gpu
    stream.memcpy_htod(&chars_vec_ram_bytes, &mut char_list_dram).expect("can't allocate the wordlist in the DRAM");
    stream.memcpy_htod(&range_vec_ram, &mut ranges_dram).expect("can't allocate the wordlist in the DRAM");
    stream.synchronize().expect("can't synchronize the threads gpu");

    // adapting the launcher to the maximum number of threads per block
    let mut mxthread_per_block: i32;
    unsafe {
        mxthread_per_block = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
    }
    
    let mut used_blocks = 1;
    if used_threads > mxthread_per_block as usize{
        // if the threads are more than the maximum allowed, split the threads in blocks, checking the occupancy
        mxthread_per_block = calulate_optimal_block_size();
        used_blocks = used_threads.div_ceil(mxthread_per_block as usize);
        used_threads = mxthread_per_block as usize;
    }
    
    let launcher = LaunchConfig {
        grid_dim: (used_blocks as u32, 1, 1),
        block_dim: (used_threads as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    
    if verbose >= 1 {
        println!("launching: {used_threads} threads per block\nlaunching: {used_blocks} blocks");
    }
    // launch the kernel
    let mut launch_args = stream.launch_builder(&fnct);
    launch_args.arg(&mut char_list_dram);
    launch_args.arg(&chrlen);
    launch_args.arg(&mut ranges_dram);
    launch_args.arg(&used_threads);
    launch_args.arg(&output_dram);
    unsafe {
        launch_args.launch(launcher).expect("can't launch the kernel");
    }

    stream.synchronize().expect("can't synchronize the threads gpu");

    if let Some(ref output) = outputfile {
        let _ = File::create(output).expect("cant create file");
    }
    
}

pub fn cuda_crack_hash(chars: String, mnlength: u64, mxlength: u64, mxcudathreads: u64, mut algo: String, verbose: u8, hash: String, hashfile: String, wordlist: String, nofile: bool, expand: bool) {
    
    // initializing the vector of hashes
    let mut hash_vec_ram: Vec<String> = vec![hash.to_string()];
    let mut hash_algo_vec_ram: Vec<u8> = Vec::new();
    let mut lines = String::new();

    if &hash == "" { // if the single hash arg is not provided, load the hashes from the file
        let mut filereader = File::open(hashfile).expect("can't open file");
        filereader.read_to_string(&mut lines).expect("can't read file");
        hash_vec_ram = lines.lines().map(|x| x.to_string()).collect(); // the hash in a vector
    }

    if !nofile {
        hash_vec_ram = hash_utils::check_if_already_found(hash_vec_ram.clone());
        if hash_vec_ram.is_empty() {
            return;
        }
    }

    algo = algo.to_lowercase();
    for entry in &hash_vec_ram {
        let vec_algo: String;
        if algo == "" {
            vec_algo = hash_utils::detect_hash(entry);
            println!("rilevated: {}", vec_algo);
        }
        else if algo == "md5" || algo == "sha-1" || algo == "sha-224" || algo == "sha-256" || algo == "sha-384" || algo == "sha-512" {
            vec_algo = algo.clone();
        }
        else {
            panic!("invalid hash!");
        }
        match vec_algo.as_str() { // assign a unique ID to each hash type for easier identification
            "md5" => hash_algo_vec_ram.push(0),
            "sha-1" => hash_algo_vec_ram.push(1),
            "sha-224" => hash_algo_vec_ram.push(2),
            "sha-256" => hash_algo_vec_ram.push(3),
            "sha-384" => hash_algo_vec_ram.push(4),
            "sha-512" => hash_algo_vec_ram.push(5),
            _ => {}
        }
    }

    let ctx = CudaContext::new(0).expect("can't create cuda context");
    let stream = ctx.default_stream();
    let mut output_found= stream.alloc_zeros::<u8>(0).unwrap();

    if wordlist == "" { // if no wordlist is provided, execute a pure-bruteforce algorithm 

        // calculate the maximum threads for chunking
        let mut mxthreads: u64;
        unsafe {
            let tot_sm = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap(); 
            let sm_mxthread = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR).unwrap(); 
            mxthreads = (tot_sm * sm_mxthread) as u64;
        }
        mxthreads = (mxthreads * mxcudathreads) / 100; // usable amount of threads (as a percentage)
        
        let chars: Vec<char> = chars.chars().collect();
        let chars: String = String::from_iter(chars);

        let ret = range_builder(&chars, mxthreads, mnlength, mxlength); // building the range and the minor values
        let total_act = ret.0;
        let act_for_thread = ret.1;
        let range_vec_ram = ret.2;
        let _ = ret;

        // starting the kernel with cuda
        let ptx = ctx.load_module(Ptx::from_file("src/gpuKernel/pure_brute.ptx")).expect("error while loading the module, be sure the module 'src/gpuKernel/WordlistBrute.ptx' exist");
        let fnct = ptx.load_function("crack_word").unwrap();

        // converting in bytes
        let hash_vec_rambytes = hash_vec_ram.join("").as_bytes().to_vec();
        let chars_vec_ram_bytes = chars.as_bytes().to_vec();
        let tothashes = hash_vec_ram.len();
        let n = range_vec_ram.len();
        let mut used_threads = range_vec_ram.len() / 3;
        let chrlen = chars.len();
        
        if verbose >= 1 {
            let mut total_used_mem = 0;
            total_used_mem += std::mem::size_of::<Vec<u8>>() + hash_vec_rambytes.capacity() * std::mem::size_of::<u8>();
            total_used_mem += std::mem::size_of::<Vec<u8>>() + hash_algo_vec_ram.capacity() * std::mem::size_of::<u8>();
            total_used_mem += std::mem::size_of::<Vec<u8>>() + chars_vec_ram_bytes.capacity() * std::mem::size_of::<u8>();
            total_used_mem += std::mem::size_of::<Vec<u64>>() + range_vec_ram.capacity() * std::mem::size_of::<u64>();
            if !nofile {
                total_used_mem += std::mem::size_of::<Vec<u64>>() + ((hash_vec_rambytes.len() / hash_vec_ram.len()) * mxlength as usize) * std::mem::size_of::<u64>();
            }
            let (meminfo1, _) = cudarc::driver::result::mem_get_info().unwrap();
            println!("memory required: {total_used_mem} bytes\nusable memory of the gpu: {meminfo1} bytes\ntotal actions to perform: {total_act}\ntotal actions to perform for each thread: {act_for_thread}");
        }

        // loading the memory
        if !nofile { // if not disabled, alloc '0' to create an output buffer
            output_found = stream.alloc_zeros::<u8>((hash_vec_rambytes.len() / hash_vec_ram.len()) * mxlength as usize).expect("can't allocate the outputVector in the DRAM");    
        }
        let mut hash_list_dram = stream.alloc_zeros::<u8>(hash_vec_rambytes.len()).expect("can't allocate the hash list in the DRAM");
        let mut hash_algo_dram = stream.alloc_zeros::<u8>(hash_algo_vec_ram.len()).expect("can't allocate the hash algo list in the DRAM");
        let mut char_list_dram = stream.alloc_zeros::<u8>(chars_vec_ram_bytes.len()).expect("can't allocate the chars list in the DRAM");
        let mut ranges_dram = stream.alloc_zeros::<u64>(range_vec_ram.len()).expect("can't allocate the ranges list in the DRAM");

        // allocate the memory in the gpu
        stream.memcpy_htod(&hash_vec_rambytes, &mut hash_list_dram).expect("can't allocate the hashes in the DRAM");
        stream.memcpy_htod(&hash_algo_vec_ram, &mut hash_algo_dram).expect("can't allocate the hashes algos in the DRAM");
        stream.memcpy_htod(&chars_vec_ram_bytes, &mut char_list_dram).expect("can't allocate the charlist in the DRAM");
        stream.memcpy_htod(&range_vec_ram, &mut ranges_dram).expect("can't allocate the rangelist in the DRAM");
        stream.synchronize().expect("can't synchronize the threads gpu");

        // adapting the launcher to the maximum number of threads per block
        let mut mxthread_per_block: i32;
        unsafe {
            mxthread_per_block = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
        }
        
        let mut used_blocks = 1;
        if used_threads > mxthread_per_block as usize{
            // if the threads are more than the maximum allowed, split the threads in blocks, checking the occupancy
            mxthread_per_block = calulate_optimal_block_size();
            used_blocks = used_threads.div_ceil(mxthread_per_block as usize);
            used_threads = mxthread_per_block as usize;
        }

        // launch the kernel        
        let launcher = LaunchConfig {
            grid_dim: (used_blocks as u32, 1, 1),
            block_dim: (used_threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut verbose_gpu = false;
        if verbose >= 1 {
            println!("launching: {used_threads} threads per block\nlaunching: {used_blocks} blocks");
            if verbose >= 2 {
                verbose_gpu = true;
            }
        }
        let mut launch_args = stream.launch_builder(&fnct);
        launch_args.arg(&mut char_list_dram);
        launch_args.arg(&chrlen);
        launch_args.arg(&mut ranges_dram);
        launch_args.arg(&mut hash_list_dram);
        launch_args.arg(&hash_algo_dram);
        launch_args.arg(&tothashes);
        launch_args.arg(&n);
        launch_args.arg(&verbose_gpu);
        launch_args.arg(&nofile);
        launch_args.arg(&output_found);
        unsafe {
            launch_args.launch(launcher).expect("can't launch the kernel");
        }
        stream.synchronize().expect("can't synchronize the threads gpu");
    }   
    else { // else, use the wordlist for a classic wordlist-bruteforce
        let mut filereader = File::open(&wordlist).expect("can't open file");
        let mut file: Vec<u8> = Vec::new();
        filereader.read_to_end(&mut file).expect("can't read file"); // reading the file as bytes
        let file = String::from_utf8_lossy(&file); 
        let mut wordlist = file.lines().map(|x| x.to_string()).collect::<Vec<String>>();
        if expand {
            wordlist = generators::expand_wordlist(wordlist);
        }
        // starting the kernel with cuda
        let ptx = ctx.load_module(Ptx::from_file("src/gpuKernel/wordlist_brute.ptx")).expect("error while loading the module, be sure the module 'src/gpuKernel/WordlistBrute.ptx' exist");
        let fnct = ptx.load_function("crack_word").unwrap();

        // loading the memory
        let word_list_vec_ram: Vec<String> = wordlist; // all the wordlist in a vector
        let mut x = word_list_vec_ram.len();

        let mut offset_vec_ram: Vec<i32> = vec![0]; // all the offsets of the word
        let mut offset_calc: usize = 0;
        for word in &word_list_vec_ram {
            offset_calc += word.len();
            offset_calc += 1;
            offset_vec_ram.push(offset_calc as i32);
        }

        let lengths_vec_ram: Vec<i32> = word_list_vec_ram.iter().map(|x| x.len() as i32).collect(); // all the lengths

        let y: usize = hash_vec_ram.len();
        let n = x * y;
        let tothashes = hash_vec_ram.len();

        // converting in bytes
        let hash_vec_rambytes = hash_vec_ram.join("").as_bytes().to_vec();
        let bytesword_list_vec_ram = word_list_vec_ram.join("\0").as_bytes().to_vec();
        
        if !nofile {
            output_found = stream.alloc_zeros::<u8>((hash_vec_rambytes.len() / hash_vec_ram.len()) * mxlength as usize).expect("can't allocate the outputVector in the DRAM");    
        }

        if verbose >= 1 {
            let mut total_used_mem = 0;
            total_used_mem += std::mem::size_of::<Vec<u8>>() + hash_vec_rambytes.capacity() * std::mem::size_of::<u8>();
            total_used_mem += std::mem::size_of::<Vec<u8>>() + bytesword_list_vec_ram.capacity() * std::mem::size_of::<u8>();
            total_used_mem += std::mem::size_of::<Vec<i32>>() + lengths_vec_ram.capacity() * std::mem::size_of::<i32>();
            total_used_mem += std::mem::size_of::<Vec<i32>>() + offset_vec_ram.capacity() * std::mem::size_of::<i32>();
            if !nofile {
                total_used_mem += std::mem::size_of::<Vec<u64>>() + ((hash_vec_rambytes.len() / hash_vec_ram.len()) * mxlength as usize) * std::mem::size_of::<u64>();
            }
            let (meminfo1, _) = cudarc::driver::result::mem_get_info().unwrap();
            println!("memory required: {total_used_mem} bytes\nusable memory of the gpu: {meminfo1} bytes\ntotal word to check: {}", word_list_vec_ram.len());
        }

        // allocate the memory in the gpu
        let mut wordlist_vec_dram = stream.alloc_zeros::<u8>(bytesword_list_vec_ram.len()).expect("can't allocate the wordlist in the DRAM");
        let mut hash_algo_dram = stream.alloc_zeros::<u8>(hash_algo_vec_ram.len()).expect("can't allocate the hash algo list in the DRAM");
        let mut offset_vec_dram = stream.alloc_zeros::<i32>(offset_vec_ram.len()).expect("can't allocate the offset list in the DRAM");
        let mut lengths_vec_dram = stream.alloc_zeros::<i32>(lengths_vec_ram.len()) .expect("can't allocate the lengths list in the DRAM");
        let mut hash_list_dram = stream.alloc_zeros::<u8>(hash_vec_rambytes.len()).expect("can't allocate the hash list in the DRAM");

        stream.memcpy_htod(&bytesword_list_vec_ram, &mut wordlist_vec_dram).expect("can't allocate the wordlist in the DRAM");
        stream.memcpy_htod(&hash_algo_vec_ram, &mut hash_algo_dram).expect("can't allocate the hashes algos in the DRAM");
        stream.memcpy_htod(&offset_vec_ram, &mut offset_vec_dram).expect("can't allocate the offset list in the DRAM");
        stream.memcpy_htod(&lengths_vec_ram, &mut lengths_vec_dram).expect("can't allocate the length list in the DRAM");
        stream.memcpy_htod(&hash_vec_rambytes, &mut hash_list_dram).expect("can't allocate the hashes in the DRAM");

        // adapting the launcher to the maximum number of threads per block
        let mut mxthread_per_block: i32;
        unsafe {
            mxthread_per_block = cudarc::driver::result::device::get_attribute(0, cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap();
        }
        
        let mut used_blocks = 1;
        if x > mxthread_per_block as usize{
            // if the threads are more than the maximum allowed, split the threads in blocks, checking the occupancy
            mxthread_per_block = calulate_optimal_block_size();
            used_blocks = x.div_ceil(mxthread_per_block as usize);
            x = mxthread_per_block as usize;
        }
        
        // launch the kernel
        let launcher = LaunchConfig {
            grid_dim: (used_blocks as u32, 1, 1),
            block_dim: (x as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut verbose_gpu = false;
        if verbose >= 1 {
            println!("launching: {x} threads per block\nlaunching: {used_blocks} blocks");
            if verbose >= 2 {
                verbose_gpu = true;
            }
        }
        let mut launch_args = stream.launch_builder(&fnct);
        launch_args.arg(&mut wordlist_vec_dram);
        launch_args.arg(&mut offset_vec_dram);
        launch_args.arg(&mut lengths_vec_dram);
        launch_args.arg(&mut hash_list_dram);
        launch_args.arg(&hash_algo_dram);
        launch_args.arg(&tothashes);
        launch_args.arg(&n);
        launch_args.arg(&verbose_gpu);
        launch_args.arg(&nofile);
        launch_args.arg(&output_found);
        unsafe {
            launch_args.launch(launcher).expect("can't launch the kernel");
        }

        stream.synchronize().expect("can't synchronize the threads gpu");

    }

    if !nofile { // after thread synchronization, update 'found' file with new found entries
        let output_vec_ram = stream.memcpy_dtov(&output_found).unwrap(); // getting the output vector from the DRAM
        let otpstr = String::from_utf8(output_vec_ram).unwrap();
        let otpstr = otpstr.split("\0");
        let mut results: Vec<&str> = Vec::new(); // this vector have this structure: <hash, word>
        for i in otpstr{
            if !i.is_empty(){ // skip empty strings that may result from splitting
                results.push(i);
            }
        }

        // rewriting the file "found" with the new entries
        let mut file = File::open("src/found").expect("can't find file 'found'");
        let mut foundfile = String::new();
        let _ = file.read_to_string(&mut foundfile);
        let mut iter = 0;
        for _ in 0..(results.len()/2) {
            if !foundfile.contains(results[iter]){ // to avoid duplication, check if it is not already present
                foundfile.push_str(format!("{}:{}\n", results[iter], results[iter+1]).as_str()); // load every entry with te format: 'HASH:WORD'
            }
            iter += 2;
        }
        let mut file = File::create("src/found").expect("can't recreate file: 'found'");
        let _ = file.write_all(foundfile.as_bytes());
    }
}
