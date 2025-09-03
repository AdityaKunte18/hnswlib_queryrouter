# hnswlib_queryrouter

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

clang++ -std=c++20 -O2 -Wall -Wextra -I. utils/utils.cpp QueryRouter.cpp main.cpp -o main -lz

### Run
./main
