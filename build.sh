cmake -G Ninja -S . -B build-gcc -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build-gcc --config Release -j
