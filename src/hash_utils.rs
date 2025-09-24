// Copyright 2025 Hakil
// Licensed under the Apache License, Version 2.0

// importing libraries
use std::{
    fs::{
        exists, 
        File
    }, 
    io::Read
};
use md5;
use sha1::{
    Sha1, 
    Digest
};
use sha2::{
    Sha224,
    Sha256,
    Sha384,
    Sha512
};

pub fn check_if_already_found(hashes: Vec<String>) -> Vec<String> {

    if !exists("src/found").unwrap(){ 
        File::create("src/found").expect("can't create 'found' file");
    }

    // opening the file "found"
    let mut file = File::open("src/found").expect("can't find file 'found'"); 
    let mut foundfile = String::new();
    let _ = file.read_to_string(&mut foundfile);
    
    // building the new hashes vector(without the already found hashes) 
    let mut returned_hashes: Vec<String> = Vec::new();
    for hash in hashes {
        let mut found = false;
        if !hash.is_empty() {
            for line in foundfile.lines() {
                if line.starts_with(&hash){
                    println!("hash in 'found' '{}' -> '{}'", hash, line.replace(&hash, "").replace(":", ""));
                    found = true;
                    continue;
                }
            }
            if !found {
                returned_hashes.push(hash);
            }
        }
    }
    returned_hashes

}

pub fn detect_hash(hash: &str) -> String{
    if hash.chars().all(|c| c.is_digit(16)) {
        if hash.len() == 32 {
            return "md5".to_string();
        }
        else if hash.len() == 40 {
            return "sha-1".to_string();
        }
        else if hash.len() == 56 {
            return "sha-224".to_string();
        }
        else if hash.len() == 64 {
            return "sha-256".to_string();
        }
        else if hash.len() == 96 {
            return "sha-384".to_string();
        }
        else if hash.len() == 128 {
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

pub fn compare_hash(hash: &str, word: &str, algo: &str, verbose: u8) -> bool { 
    match algo {
        "md5" => {
            if format!("{:x}",md5::compute(&word)) == hash {
                println!("FOUND MATCH: '{}' -> '{}'", hash, word);
                return true;
            }
        }, 
        "sha-1" => {
            let mut sha1_hasher = Sha1::new();
            sha1_hasher.update(&word);
            let sha_result = sha1_hasher.finalize();
            if format!("{:x}", sha_result) == hash {
                println!("FOUND MATCH: '{}' -> '{}'", hash, word);
                return true;
            }
        },
        "sha-224" => {
            let mut sha224_hasher = Sha224::new();
            sha224_hasher.update(&word);
            let sha_result = sha224_hasher.finalize();
            if format!("{:x}", sha_result) == hash {
                println!("FOUND MATCH: '{}' -> '{}'", hash, word);
                return true;
            }
        },
        "sha-256" => {
            let mut sha256_hasher = Sha256::new();
            sha256_hasher.update(&word);
            let sha_result = sha256_hasher.finalize();
            if format!("{:x}", sha_result) == hash {
                println!("FOUND MATCH: '{}' -> '{}'", hash, word);
                return true;
            }
        },
        "sha-384" => {
            let mut sha384_hasher = Sha384::new();
            sha384_hasher.update(&word);
            let sha_result = sha384_hasher.finalize();
            if format!("{:x}", sha_result) == hash {
                println!("FOUND MATCH '{}' -> '{}'", hash, word);
                return true;
            }
        },
        "sha-512" => {
            let mut sha512_hasher = Sha512::new();
            sha512_hasher.update(&word);
            let sha_result = sha512_hasher.finalize();
            if format!("{:x}", sha_result) == hash {
                println!("FOUND MATCH: '{}' -> '{}'", hash, word);
                return true;
            }
        },
        _ => {}
    }
    if verbose == 2 {
        println!("dont work: '{}'", word);
    }
    return false;

}
