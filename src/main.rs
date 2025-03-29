use clap::{builder::Str, Parser};
use core::hash;
use std::fmt::format;
use std::{fs::File, string};
use std::io::{self, Read, Write};
use md5;
use sha1::{Sha1, Digest};
use sha2::{Sha256, Sha512};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[derive(Parser, Debug)]
#[command(about = "RustCrack can crack the hashes of MD5, SHA-1, SHA-256 and SHA-512 or generate simple wordlists", long_about = None)]

struct Args {
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

}
fn main(){
    let args = Args::parse();
    if args.generatewordlist == true{
        generatewordlist(args.chars.clone(), args.mnlenght, args.mxlenght, args.outputfile);
    }
    if args.crackhash == true{
        crackhash(args.chars.clone(), args.mnlenght, args.mxlenght, args.algorithm.unwrap_or_default(), args.verbose, args.hash.unwrap_or_default(), args.hashfile.unwrap_or_default(), args.wordlist.unwrap_or_default());
    }
}

fn crackhash(chars: String, mnlenght: u64, mxlenght: u64, mut algo: String, verbose: bool, hash: String, hashfile: String, wordlist: String){
    if hash != ""{
        algo = algo.to_lowercase();
        if algo == ""{
            algo = detecthash(hash.clone());
            println!("rilevated: {}", algo);
        }
        else if algo == "md5" || algo == "sha-1" || algo == "sha-256" || algo == "sha-512" {
            algo = algo;
        }
        else {
            panic!("invalid hash!")
        }
        if wordlist == ""{
            let chars: Vec<char> = chars.chars().collect();
                for length in mnlenght..=mxlenght {
                    let total = (chars.len() as u64).pow(length as u32);
                    for n in 0..total {
                        let mut word = String::new();
                        let mut temp = n;
                        for _ in 0..length {
                            word.push(chars[(temp % chars.len() as u64) as usize]);
                            temp /= chars.len() as u64;
                        }
                        comparehash(hash.clone(), word.clone(), algo.clone(), verbose.clone());
                }
            }
        }
        else {
            let mut filereader = File::open(wordlist.clone()).expect("cant open file");
            let mut file = String::new();
            filereader.read_to_string(&mut file);
            for word in file.lines(){
                comparehash(hash.clone(), word.clone().to_string(), algo.clone(), verbose.clone());
            }
        }
    }
    else {
        let chars: Vec<char> = chars.chars().collect();
        let mut filereader = File::open(hashfile).expect("cant open file");
        let mut file = String::new();
        filereader.read_to_string(&mut file);
        for hash in file.lines(){
            println!("trying '{}'", hash);
            algo = detecthash(hash.clone().to_string());
            println!("rilevated: {}", algo);
            if wordlist == ""{
                let mut found = false; 
                for length in mnlenght..=mxlenght {
                    let total = (chars.len() as u64).pow(length as u32);
                    for n in 0..total {
                        let mut word = String::new();
                        let mut temp = n;
                        for _ in 0..length {
                            word.push(chars[(temp % chars.len() as u64) as usize]);
                            temp /= chars.len() as u64;
                        }
                        if comparehash(hash.clone().to_string(), word.clone(), algo.clone(), verbose.clone()){
                            found = true;
                            break;
                        }
                    }
                    if found{
                        break;
                    }
                }
            }
            else{
                let mut filereader = File::open(wordlist.clone()).expect("cant open file");
                let mut file = String::new();
                filereader.read_to_string(&mut file);
                for word in file.lines(){
                    if comparehash(hash.clone().to_string(), word.clone().to_string(), algo.clone(), verbose.clone()){
                        break;
                    }
                }
            }
        }
    }

}

fn generatewordlist(chars: String, mnlenght: u64, mxlenght: u64, outputfile: Option<String>) {
    let mut filewriter: Option<File> = None;
    if let Some(ref output) = outputfile {
        filewriter = Some(File::create(output).expect("cant create file"));
    }
    let chars: Vec<char> = chars.chars().collect();
    for length in mnlenght..=mxlenght {
        let total = (chars.len() as u64).pow(length as u32);
        for n in 0..total {
            let mut word = String::new();
            let mut temp = n;
            for _ in 0..length {
                word.push(chars[(temp % chars.len() as u64) as usize]);
                temp /= chars.len() as u64;
            }
            if let Some(ref mut writer) = filewriter {
                word.push_str("\n");
                writer.write_all(word.as_bytes()).expect("Failed to write to file");
            } else {
                println!("{}", word);
            }

        }

    }

}

fn detecthash(hash: String) -> String{
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
    if algo == "md5" && format!("{:x}",md5::compute(word.clone())) == hash.clone(){
        println!("FOUND MATCH: '{}' -> '{}'", hash.clone(), word.clone());
        return true;
    }
    else if algo == "sha-1"{
        let mut sha1_hasher = Sha1::new();
        sha1_hasher.update(word.clone());
        let sha_result = sha1_hasher.finalize();
        if format!("{:x}", sha_result) == hash.clone(){
            println!("FOUND MATCH: '{}' -> '{}'", hash.clone(), word.clone());
            return true;
        }
    }
    else if algo == "sha-256"{
        let mut sha256_hasher = Sha256::new();
        sha256_hasher.update(word.clone());
        let sha_result = sha256_hasher.finalize();
        if format!("{:x}", sha_result) == hash.clone(){
            println!("FOUND MATCH: '{}' -> '{}'", hash.clone(), word.clone());
            return true;
        }
    }
    else if algo == "sha-512"{
        let mut sha512_hasher = Sha512::new();
        sha512_hasher.update(word.clone());
        let sha_result = sha512_hasher.finalize();
        if format!("{:x}", sha_result) == hash.clone(){
            println!("FOUND MATCH: '{}' -> '{}'", hash.clone(), word.clone());
            return true;
        }
    }
    if verbose{
        println!("dont work: '{}'", word);
    }
    return false;

}
