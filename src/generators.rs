// Copyright 2025 Hakil
// Licensed under the Apache License, Version 2.0

// importing modules
use crate::hash_utils;

// importing libraries
use std::{ 
    collections::HashMap, fs::File, io::{
        Read, 
        Write
    }
};

pub fn generatewordlist(chars: String, mnlength: u64, mxlength: u64, outputfile: Option<String>, verbose: u8) { // same algorithm of the pure-bruteforcing
    let mut filewriter: Option<File> = None;
    if let Some(ref output) = outputfile {
        filewriter = Some(File::create(output).expect("cant create file"));
    }
    let chars: Vec<char> = chars.chars().collect();
    
    for length in mnlength..=mxlength { // Loop through word lengths from minimum to maximum
        let total = (chars.len() as u64).pow(length as u32); // Total number of possible combinations for this word length, chars * length create a matrix of possibilities
        for n in 0..total {
            let mut word = String::new();
            let mut temp = n;
            for _ in 0..length { // Generate each character of the word based on current number
                word.push(chars[(temp % chars.len() as u64) as usize]); // insert in word the expected char(based on the number of position of possibility)
                temp /= chars.len() as u64; // temp = temp / chars length
            }
            if let Some(ref mut writer) = filewriter {
                word.push_str("\n");
                writer.write_all(word.as_bytes()).expect("Failed to write to file");
            } 
            if outputfile == None || verbose == 2 {
                println!("{}", word.replace("\n", ""));                
            }
        }
    }
}

pub fn crackhash(chars: String, mnlength: u64, mxlength: u64, input_algo: String, verbose: u8, hash: String, hashfile: String, wordlist: String, nofile: bool, expand: bool) {
  
    // initializing the vector of hashes
    let mut hash_vec_ram: Vec<&str> = vec![&hash];
    let mut hashfound: Vec<String> = Vec::new();
    let mut lines = String::new();

    if &hash == "" { // if the single hash arg is not provided, load the hashes from the file
        let mut filereader = File::open(hashfile).expect("can't open file");
        filereader.read_to_string(&mut lines).expect("can't read file");
        hash_vec_ram = lines.lines().collect(); // the hash in a vector
    }

    if !nofile {
        hash_vec_ram = hash_utils::check_if_already_found(hash_vec_ram.clone());
    }
    let chars: Vec<char> = chars.chars().collect();
    
    for hash in &hash_vec_ram {
        println!("trying '{}'", hash);
        let mut algo = input_algo.to_lowercase();
        if algo == "" {
            if *hash != "" {
                algo = hash_utils::detect_hash(hash); 
            }
            else {
                algo = hash_utils::detect_hash(&hash_vec_ram[0]);
            }
            println!("rilevated: {}", algo);
        }
        else if algo == "md5" || algo == "sha-1" || algo == "sha-224" || algo == "sha-256" || algo == "sha-384" || algo == "sha512" {
            algo = algo
        }
        else {
            panic!("invalid hash!");
        }

        if &wordlist == "" { // if no wordlist is provided, execute a pure-bruteforce algorithm
            let mut found = false;
            for length in mnlength..=mxlength { // Loop through word lengths from minimum to maximum
                let total = (chars.len() as u64).pow(length as u32); // Total number of possible combinations for this word length, chars * length create a matrix of possibilities
                for n in 0..total {
                    let mut word = String::new();
                    let mut temp = n;
                    for _ in 0..length { // Generate each character of the word based on current number
                        word.push(chars[(temp % chars.len() as u64) as usize]); // insert in word the expected char(based on the number of position of possibility)
                        temp /= chars.len() as u64; // temp = temp / chars length
                    }
                    if hash_utils::compare_hash(hash, &word, &algo, verbose){ 
                        found = true;
                        if !nofile {
                            hashfound.push(format!("{}:{}\n", hash, word));
                        }
                        break;
                    }
                }
                if found{
                    break;
                }
            }
        }
        else { // else, use the wordlist for a classic wordlist-bruteforce
            let mut filereader = File::open(&wordlist).expect("can't open file"); // loading the wordlist
            let mut file = String::new();
            filereader.read_to_string(&mut file).expect("can't read file");
            let mut wordlist = file.lines().map(|x| x.to_string()).collect::<Vec<String>>();
            if expand {
                wordlist = expand_wordlist(wordlist);
            }
            for word in wordlist { // do the brute-force
                if hash_utils::compare_hash(hash, &word, &algo, verbose){
                    if !nofile {
                        hashfound.push(format!("{}:{}\n", hash, word));
                    }
                    break;
                }
            }
        }

        if !nofile {
            // rewriting the file "found" with the new entries
            let mut file = File::open("src/found").expect("can't find file 'found'");
            let mut foundfile = String::new();
            let _ = file.read_to_string(&mut foundfile);
            for find in &hashfound {
                if !foundfile.contains(find){ // to avoid duplication, check if it is not already present
                    foundfile.push_str(&find);
                }
            }
            let mut file = File::create("src/found").expect("can't recreate file: 'found'");
            let _ = file.write_all(foundfile.as_bytes());
        }   
    }   
}

pub fn expand_wordlist(wordlist: Vec<String>) -> Vec<String> {
    
    let mut new_wordlist: Vec<String> = Vec::new();
    let trasf_char = vec!['a', 'c', 'e', 'g', 'i', 'l', 'o', 'r', 's', 't', 'u', 'y']; // these are the characters that are subject to variation

    let mut char_options: HashMap<char, Vec<char>> = HashMap::new(); // these are the variation of the characters 
    char_options.insert('a', vec!['a', '4', 'à', '@']);
    char_options.insert('c', vec!['c', '(', 'ç']);
    char_options.insert('e', vec!['e', '3', '£']);
    char_options.insert('g', vec!['g', '9']);
    char_options.insert('i', vec!['i', '1', '!', 'ì', '/', '|']);
    char_options.insert('l', vec!['l', '7']);
    char_options.insert('o', vec!['o', '0', 'ò']);
    char_options.insert('r', vec!['r', '4']);
    char_options.insert('s', vec!['s', '2', '5', '$', '§']);
    char_options.insert('t', vec!['t', '7']);
    char_options.insert('u', vec!['u', 'ù']);
    char_options.insert('y', vec!['y', '7']);

    for word in wordlist {
        let mut changeable_chars: Vec<char> = Vec::new();
        for c in word.chars() {
            if trasf_char.contains(&c) {
                changeable_chars.push(c);
            }
        }
        new_wordlist.extend(generate_possibilities_tree(vec![word], changeable_chars, &char_options));
    }

    new_wordlist
}

fn generate_possibilities_tree(wordlist: Vec<String>, mut changeable_chars: Vec<char>, char_options: &HashMap<char, Vec<char>>) -> Vec<String> { // this function uses recursion to generate all the possibilities, it works in a tree-like way.
    
    let mut new_wordlist: Vec<String> = Vec::new();
    let char_to_change = changeable_chars[0];
    let total_variations = char_options.get(&char_to_change).unwrap().len();

    for entry in wordlist {
        for idx in 0..total_variations {
            new_wordlist.push(entry.replacen(&char_to_change.to_string(), &char_options.get(&char_to_change).unwrap()[idx].to_string(), 1).to_owned());
        }
    }

    changeable_chars.remove(0); // removing the first changed character

    if changeable_chars.len() != 0 { // if there are more chars to change, repeat the function
        new_wordlist = generate_possibilities_tree(new_wordlist.clone(), changeable_chars, char_options);
    }
    
    new_wordlist
}