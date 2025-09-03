# hnswlib_queryrouter

### Get data from http://corpus-texmex.irisa.fr/


#### for SIFT 1M (500 MB)
```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
```

#### for SIFT 1B (92 GB)
```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
```

### Build hnswlib (this lets you run the examples)

```bash
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
mkdir build
cd build
cmake ..
make all
```

### Build main.cpp, QueryRouter.cpp and utils.cpp

```bash
clang++ -std=c++20 -O2 -Wall -Wextra -I. utils/utils.cpp QueryRouter.cpp main.cpp -o main -lz
```

### Run

```
./main
```
