# INSTALL for Debian

----------

Install all the dependencies:


Part 1: Install git, Rust and the Nvidia drivers, then reboot the system.

```

sudo apt update &&

sudo apt upgrade &&

sudo apt install git &&

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh &&

sudo apt install nvidia-driver-535 &&

sudo reboot

```


Part 2: Install CUDA and export his path in ```~/.bashrc```.

```

sudo apt install cuda &&

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc &&

echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc &&

source ~/.bashrc

```


Install RustCrack:

```

git clone https://github.com/H4k1l/RustCrack.git

```

# INSTALL for Fedora

----------

Install all the dependencies:


Part 1: Install git, Rust and the Nvidia drivers, then reboot the system.

```

sudo dnf update &&

sudo dnf install git &&

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh &&

sudo dnf install akmod-nvidia &&

sudo reboot

```


Part 2: Install CUDA and export his path in ```~/.bashrc```.

```

sudo dnf install xorg-x11-drv-nvidia-cuda &&

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc &&

echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc &&

source ~/.bashrc

```
  

Install RustCrack:

```

git clone https://github.com/H4k1l/RustCrack.git

```

# INSTALL for Arch

----------

Install all the dependencies:


Part 1: Install git, Rust and the Nvidia drivers, then reboot the system.

```

sudo pacman -Syu &&

sudo pacman -S git &&

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh &&

sudo pacman -S nvidia &&

sudo reboot

```


Part 2: Install CUDA and export his path in ```~/.bashrc```.

```

sudo pacman -S cuda &&

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc &&

echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc &&

source ~/.bashrc

```


Install RustCrack:

```

git clone https://github.com/H4k1l/RustCrack.git

```

# INSTALL for Windows

----------

Install all the dependencies:


Part 1: Install git, Rust and the Nvidia drivers, then reboot the system.

```

winget upgrade --all;

winget install --id Git.Git -e --source winget;

Invoke-WebRequest https://win.rustup.rs/x86_64 -OutFile rustup-init.exe; Start-Process .\rustup-init.exe;

```

At this point, you need to install the Nvidia drivers from [this](https://www.nvidia.com/Download/index.aspx) web page and reboot the system.


Part 2: Install CUDA.


You need to install the Nvidia drivers from [this](https://developer.nvidia.com/cuda-downloads) web page and reboot the system.


Part 3: export the CUDA path.

```

setx PATH "$Env:PATH;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin";

setx CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5";

```


Install RustCrack:

```

git clone https://github.com/H4k1l/RustCrack.git;

```
