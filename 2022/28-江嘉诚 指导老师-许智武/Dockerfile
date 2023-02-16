FROM ubuntu:22.04
COPY . /home/CollAFL-2.57b
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak \
    && sed -i "s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g" /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update --fix-missing \
    && apt-get install -y wget sudo build-essential libacl1-dev --fix-missing \

    && apt-get install -y clang-11 llvm-11 --fix-missing \
    && ln -s /usr/bin/clang-11 /usr/bin/clang \
    && ln -s /usr/bin/clang++-11 /usr/bin/clang++ \
    && ln -s /usr/bin/llvm-ar-11 /usr/bin/llvm-ar \
    && ln -s /usr/bin/llvm-as-11 /usr/bin/llvm-as \
    && ln -s /usr/bin/llvm-config-11 /usr/bin/llvm-config \

    && cd /home/CollAFL-2.57b \
    && make \
    && cd /home/CollAFL-2.57b/llvm_mode \
    && make \
    && cd /home/CollAFL-2.57b \
    && make install \
