#!bin/bash
set -e
wget http://panda.moyix.net/~moyix/lava_corpus.tar.xz 
tar -xvf lava_corpus.tar.xz 
rm lava_corpus.tar.xz  
cd lava_corpus/LAVA-M/base64/coreutils-8.24-lava-safe  
sed -i 's/IO_ftrylockfile/IO_EOF_SEEN/' lib/*.c  
echo "#define _IO_IN_BACKUP 0x100" >> lib/stdio-impl.h 
cd .. 
sed -i '1i#include <sys/sysmacros.h>' coreutils-8.24-lava-safe/lib/mountlist.c 
sed -i 's/.\/configure/.\/configure FORCE_UNSAFE_CONFIGURE=1/g' ./validate.sh 
export CC=afl-clang-fast
export CXX=afl-clang-fast++
./validate.sh 
afl-fuzz -m none -t 5000 -i fuzzer_input/ -o outputs coreutils-8.24-lava-safe/lava-install/bin/base64 -d @@