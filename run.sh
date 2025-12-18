rm -rf build
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/libomp ..
make
./hss_test