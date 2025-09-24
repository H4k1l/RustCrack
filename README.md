# RustCrack
----
RustCrack can crack the hashes of MD5, SHA-1, SHA-256 and SHA-512 or generate simple wordlists. It can also use CUDA to parallelize the tasks. RustCrack is designed to be efficent, fast and reliable.

# Quick Install
----
For more detailed installation instructions, refer to [INSTALL.md](./docs/INSTALL.md).

## Install for Debian
Part 1:
```
sudo apt update && sudo apt upgrade && sudo apt install git && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh & sudo apt install nvidia-driver-535 && sudo reboot
```
Part 2:
```
sudo apt install cuda && echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc
```
Install RustCrack:
```
git clone https://github.com/H4k1l/RustCrack.git
```

## Install for Fedora
Part 1:
```
sudo dnf update && sudo dnf install git && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && sudo dnf install akmod-nvidia && sudo reboot
```
Part 2:
```
sudo dnf install xorg-x11-drv-nvidia-cuda && echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc
```
Install RustCrack:
```
git clone https://github.com/H4k1l/RustCrack.git
```

## Install for Arch
Part 1:
```
sudo pacman -Syu && sudo pacman -S git && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && sudo pacman -S nvidia && sudo reboot
```
Part 2:
```
sudo pacman -S cuda && echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && source ~/.bashrc
```
Install RustCrack:
```
git clone https://github.com/H4k1l/RustCrack.git
```

## Install for Windows
Part 1:
```
winget upgrade --all; winget install --id Git.Git -e --source winget;Invoke-WebRequest https://win.rustup.rs/x86_64 -OutFile rustup-init.exe; Start-Process .\rustup-init.exe;
```
At this point, you need to install the Nvidia drivers from [this](https://www.nvidia.com/Download/index.aspx) web page and reboot the system.
Part 2:
Now you need to install the Nvidia drivers from [this](https://developer.nvidia.com/cuda-downloads) web page and reboot the system.
Part 3:
```
setx PATH "$Env:PATH;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"; setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5";
```
Install RustCrack:
```
git clone https://github.com/H4k1l/RustCrack.git
```

# Screenshots
----
![RustCrack](https://github.com/H4k1l/RustCrack/blob/main/images/screenshot1.png)
![RustCrack](https://github.com/H4k1l/RustCrack/blob/main/images/screenshot2.png)
# Usage
----
```
cargo run -- -h
RustCrack v1.1.2 can crack the hashes of MD5, SHA-1, SHA-224, SHA-256, SHA-384 and SHA-512 or generate simple wordlists. It can also use CUDA to parallelize the tasks. RustCrack is designed to be efficent, fast and reliable.

Usage: RustCrack [OPTIONS]

Options:
  -c, --chars <CHARS>                  [default: !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~]
  -m, --mnlenght <MNLENGHT>            [default: 1]
  -x, --mxlenght <MXLENGHT>            [default: 10]
  -o, --output <OUTPUTFILE>            
      --gen                            generate a wordlist
  -a, --algo <ALGORITHM>               the hash algorithm to use (md5/sha-1/sha-224/sha-256/sha-384/sha-512)
  -w, --wordlist <WORDLIST>            try to use a wordlist to brute the hash
  -H, --hash <HASH>                    
      --hashfile <HASHFILE>            
  -v, --verbose...                     use -vv to get a more verbose output
      --crack                          crack a hash
  -t, --mxcudathreads <MXCUDATHREADS>  maximum usable of cuda threads in percentage [default: 100]
      --gpu                            use CUDA to maximize the efficency for nvidia GPU's
      --nofile                         don't check if the hash is already found, and don't save the hash result
      --expand                         expands the wordlist with new possible password combinations(example: 'hello' -> 'h3ll0')
  -h, --help                           Print help
```
For crack an hash using a pure brute-force alorithm, you can simply run this command:
```
cargo run -- --crack -H <HASH>
```
You can personalize the process with various flag:
```
cargo run -- --crack --hashfile <HASHFILE> -a <ALGORITHM> -c abcdefghijklmnopqrstuvwxyz 
```
If you provide a wordlist file RustCrack will use it to brute the hash, with this method, the use of the flags: "-c", "-x" and "-m" is useless.
```
cargo run ----crack -H <HASH> -w <WORDLIST>
```
RustCrack provide the CUDA GPU kernels for all of his original algorithms, you can access them via the --gpu flag:
```
cargo run ----crack -H <HASH> --gpu
```
Verbose mode when using GPU kernels is very useful, as it provides information about the cracking process (e.g., memory usage in DRAM), you can enable it with the flag "-v":
```
cargo run -- --crack -H <HASH> --gpu -v
```
Since the algorithm of the wordlist generator is the same as that of pure brute force, the parameters are the same:
```
cargo run -- --gen -o <OUTPUTFILE> -c abcdefghijklmnopqrstuvwxyz -x 8 -m 3
```
By default RustCrack stores the cracked hashes in a file to avoid having to crack them again, this storage and control action is bypassable with the flag "--nofile":
```
cargo run -- --crack -H <HASH> --nofile
```
# Disclaimers
----
The author is not responsible for any damages, misuse or illegal activities resulting from the use of this code.

# LICENSE
----
This project is distributed under license [Apache 2.0](LICENSE).
