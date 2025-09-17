// Copyright 2025 Hakil
// Licensed under the Apache License, Version 2.0

// importing modules
mod generators;
mod cuda_handler;
mod hash_utils;

// importing libraries
use clap::{ArgAction, Parser};

#[derive(Parser, Debug)]
#[command(about = "RustCrack v1.1.2 can crack the hashes of MD5, SHA-1, SHA-224, SHA-256, SHA-384 and SHA-512 or generate simple wordlists. It can also use CUDA to parallelize the tasks. RustCrack is designed to be efficent, fast and reliable.", long_about = None)]

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

    #[clap(short = 'a', long = "algo", help = "the hash algorithm to use (md5/sha-1/sha-224/sha-256/sha-384/sha-512)")]
    algorithm: Option<String>,

    #[clap(short = 'w', long, help = "try to use a wordlist to brute the hash")]
    wordlist: Option<String>,

    #[clap(short = 'H', long = "hash")]
    hash: Option<String>,

    #[clap(long = "hashfile")]
    hashfile: Option<String>,

    #[clap(short = 'v', long = "verbose", action = ArgAction::Count, help = "use -vv to get a more verbose output")]
    verbose: u8,
    
    #[clap(long = "crack", help = "crack a hash")]
    crackhash: bool,

    #[clap(short = 't', long, help = "maximum usable of cuda threads in percentage", default_value_t = 100)]
    mxcudathreads: u64,

    #[clap(long = "gpu", help = "use CUDA to maximize the efficency for nvidia GPU's")]
    gpu: bool,

    #[clap(long = "nofile", help = "don't check if the hash is already found, and don't save the hash result", default_value_t = false)]
    nofile: bool,

    #[clap(long = "expand", help = "expands the wordlist with new possible password combinations(example: 'hello' -> 'h3ll0')")]
    expand: bool

}

fn main(){
    if env::args().len() == 1 {
        println!("No arguments provided.\nFor more information, try '--help'.");
    }
    let args = Args::parse();
    if args.generatewordlist{
        if args.gpu {
            cuda_handler::cuda_generate_wordlist(args.chars.to_owned(), args.mnlenght, args.mxlenght, args.outputfile.clone(), args.mxcudathreads, args.verbose);
        }
        generators::generatewordlist(args.chars.to_owned(), args.mnlenght, args.mxlenght, args.outputfile, args.verbose);
    }
    if args.crackhash{
        if args.gpu {
            cuda_handler::cuda_crack_hash(args.chars, args.mnlenght, args.mxlenght, args.mxcudathreads,args.algorithm.unwrap_or_default(), args.verbose, args.hash.unwrap_or_default(), args.hashfile.unwrap_or_default(), args.wordlist.unwrap_or_default(), args.nofile, args.expand);
        }
        else {
            generators::crackhash(args.chars, args.mnlenght, args.mxlenght, args.algorithm.unwrap_or_default(), args.verbose, args.hash.unwrap_or_default(), args.hashfile.unwrap_or_default(), args.wordlist.unwrap_or_default(), args.nofile, args.expand);
        }
    }
}
