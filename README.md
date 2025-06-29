# RustCrack
----
RustCrack can crack the hashes of MD5, SHA-1, SHA-256 and SHA-512 or generate simple wordlists
# Screenshots
----
![RustCrack](https://github.com/H4k1l/RustCrack/blob/main/images/screenshot1.png)
# Usage
----
  ```
cargo run -- -h
RustCrack can crack the hashes of MD5, SHA-1, SHA-256 and SHA-512 or generate simple wordlists

Usage: RustCrack [OPTIONS]

Options:
  -c, --chars <CHARS>                  [default: !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~]
  -m, --mnlenght <MNLENGHT>            [default: 1]
  -x, --mxlenght <MXLENGHT>            [default: 10]
  -o, --output <OUTPUTFILE>            
      --gen                            generate a wordlist
  -a, --algo <ALGORITHM>               the hash algorithm to use (md5/sha-1/sha-256/sha-512)
  -w, --wordlist <WORDLIST>            try to use a wordlist to brute the hash
  -H, --hash <HASH>                    
      --hashfile <HASHFILE>            
  -v, --verbose                        
      --crack                          crack a hash
      --gpu                            use CUDA to maximize the efficency for nvidia GPU's
  -t, --mxcudathreads <MXCUDATHREADS>  maximum usable of cuda threads in percentage [default: 100]
  -h, --help                           Print help
```
# Disclaimers
----
The author is not responsible for any damages, misuse or illegal activities resulting from the use of this code.

# LICENSE
----
This project is distributed under license [Apache 2.0](LICENSE).
