cmake_minimum_required(VERSION 3.10)
project(trajectory_optimization)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(Eigen3 REQUIRED)
find_package(NLopt REQUIRED)

# 设置CppAD路径
set(CPPAD_INCLUDE_DIRS "/usr/local/include")
set(CPPAD_LIBRARY_DIRS "/usr/local/lib")

# 添加可执行文件
add_executable(${PROJECT_NAME} trajectory_optimization.cpp)

# 链接库
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
    nlopt 
    pthread
    Eigen3::Eigen
    cppad_lib
)
# sudo apt-get install libcppad-dev
# 添加包含目录
target_include_directories(${PROJECT_NAME} 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CPPAD_INCLUDE_DIRS}
)

# 添加库目录
target_link_directories(${PROJECT_NAME}
    PRIVATE
    ${CPPAD_LIBRARY_DIRS}
)

# 设置编译选项
target_compile_options(${PROJECT_NAME} 
    PRIVATE 
    -O2 
    -Wall 
    -Wextra
)

# 设置运行时库搜索路径
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    INSTALL_RPATH "${CPPAD_LIBRARY_DIRS}"
    BUILD_WITH_INSTALL_RPATH TRUE
) 