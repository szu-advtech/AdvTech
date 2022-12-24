# cmake要求的最低版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名和版本
project(cloudsync VERSION 1.0)

# 指定使用的cpp版本
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 检查编译器是否支持interprocedural optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT error)
# 如果支持则开启interprocedural optimization
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
endif()
# 在Darwin或macos，这个变量可以设置为never，作用是不查找FRAMEWORK配置文件
set(CMAKE_FIND_FRAMEWORK NEVER)

# build
# 链接上阿里云sdk
SET(ALI_SOURCE_DIR /Users/liang/Downloads/aliyun-oss-cpp-sdk)
include_directories(${ALI_SOURCE_DIR}/sdk/include)
# 坑点：为什么链接静态库失败，链接动态库成功？？？
SET(ALI_LIBRARIES /Users/liang/Downloads/aliyun-oss-cpp-sdk/build/lib/libalibabacloud-oss-cpp-sdk.dylib)
link_libraries(${ALI_LIBRARIES})	
link_libraries(${CRYPTO_LIBS})
link_libraries(${CLIENT_LIBS})

# 将cryptopp放在external文件夹中，在编译的时候生成库，并指定这些库的保存位置，换台机器也能一键编译好，不需要准备好用到的库什么的，移植性好
# 开发的时候可以用下面这种做法，发布的时候可以用上面这种做法，在编译的时候生成用到的外部库存
# build的时候不生成库，还是有什么命令的？install的时候可以生成库，供别的程序使用？？那build的时候可执行文件怎么用动态库啊？？
# 而不是像上面先生成库，然后再链接库，这样做的话，移植到别的机器上的时候，需要手动安排库的组织，移植性不好

# 复制导入cryptopp的CMakeLists.txt等文件，为了使cryptopp这个外部库在cmake中能用
file(COPY ${CMAKE_SOURCE_DIR}/external/cmake-cryptopp/CMakeLists.txt
DESTINATION ${CMAKE_SOURCE_DIR}/external/cryptopp)
file(COPY ${CMAKE_SOURCE_DIR}/external/cmake-cryptopp/config.pc.in
DESTINATION ${CMAKE_SOURCE_DIR}/external/cryptopp)
file(COPY ${CMAKE_SOURCE_DIR}/external/cmake-cryptopp/cryptoppConfig.cmake
DESTINATION ${CMAKE_SOURCE_DIR}/external/cryptopp)
file(COPY ${CMAKE_SOURCE_DIR}/external/cmake-cryptopp/sources.cmake
DESTINATION ${CMAKE_SOURCE_DIR}/external/cryptopp)

# EXTERNAL_DIR变量为“external”这个目录
set(EXTERNAL_DIR external)
# 将“external”这个目录加入到头文件搜索路径中
include_directories(${EXTERNAL_DIR})
# CRYPTOPP_DIR变量为cryptopp所在目录
set(CRYPTOPP_DIR ${EXTERNAL_DIR}/cryptopp)
# 用了这条命令，子目录才会会进行编译，默认将编译后的库放在cmake输出目录下
add_subdirectory(${CRYPTOPP_DIR})
# target链接库后，target对应的cpp文件才能调用库中的头文件
link_libraries(cryptopp)

# CRYPTOPP_DIR变量为cryptopp所在目录
set(JSON_DIR ${EXTERNAL_DIR}/json)
# Typically you don't care so much for a third party library's tests to be
# run from your own project's code.
set(JSON_BuildTests OFF CACHE INTERNAL "")
# If you only include this third party in PRIVATE source files, you do not
# need to install it when your main project gets installed.
# set(JSON_Install OFF CACHE INTERNAL "")
# Don't use include(nlohmann_json/CMakeLists.txt) since that carries with it
# unintended consequences that will break the build.  It's generally
# discouraged (although not necessarily well documented as such) to use
# include(...) for pulling in other CMake projects anyways.
add_subdirectory(${JSON_DIR})
#add_library(nlohmann_json STATIC)
link_libraries(nlohmann_json::nlohmann_json)


# 注意：这些命令的执行也是有顺序的，生成用到了外部库的自己库，应该在链接外部库后再来生成链接，否则自己库会找不到外部库
# 所以最好在最后来生成target会比较好，比较能把握住顺序
# 将source文件夹加入到头文件搜索目录
include_directories(sources)
# 将满足glob expression的文件存到SOURCES这个列表中
# 因为一些h文件可能也有方法的定义，所以也要编译为库
file(
    GLOB
    SOURCES
    sources/*.cpp
    sources/*.h)
# 将SOURCES列表中的文件编译为为securefs-static静态库
# 因为使用main.cpp去生成可执行文件，并用到了sources这个子文件夹中的方法
# 所以在sources这个子文件夹中的文件要生成库并链接到main.cpp生成的可执行文件上
# 在这里进行add，就不需要在sources这个子文件夹在用一个单独的cmake文件去add
add_library(test-static STATIC ${SOURCES})
# 后面的target链接到securefs-static静态库
link_libraries(test-static)

# 配置编译器链接参数，设置warning的情况
# https://gcc.gnu.org/onlinedocs/gcc-4.3.2/gcc/Warning-Options.html
# -Wall 开启所有警告（可以理解为warinig all），使用它能够使编译器产生尽可能多的警告信息
# -Wextra 启用一些未被-Wall启用的额外警告标志。(这个选项过去被称为-W。仍然支持旧的名称，但新名称更具描述性。)
# -Wno-unknown-pragmas 禁用未知的pragma警告，一些第三方库产生的无害警告可以通过这个选项来屏蔽。（可以屏蔽cryptopp产生的无害警告）
add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
# # 关闭rtti，这样的话typeid、dynamic_cast等方法不能用了，与cryptopp起冲突
# add_compile_options(-fno-rtti )

# 生成可执行文件
add_executable(${PROJECT_NAME} main.cpp)