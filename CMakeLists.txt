cmake_minimum_required(VERSION 3.20)
project(BEC4D LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures (GTX 1070 = Compute Capability 6.1)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(USE_CUDA "Enable CUDA acceleration" ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

# For Windows: GLFW and GLAD can be installed via vcpkg
# vcpkg install glfw3 glad glm imgui[glfw-binding,opengl3-binding] hdf5
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# Optional: HDF5 for file I/O
find_package(HDF5 COMPONENTS CXX)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Source files
set(CORE_SOURCES
    src/core/math_utils.cpp
    src/core/timer.cpp
)

set(SIMULATION_SOURCES
    src/simulation/simulation_params.cpp
    src/simulation/hypersphere_bec.cpp
    src/simulation/neighbor_tree.cpp
)

set(SIMULATION_CUDA_SOURCES
    src/simulation/cuda/evolution_kernels.cu
)

set(VISUALIZATION_SOURCES
    src/visualization/renderer.cpp
    src/visualization/shader.cpp
    src/visualization/camera.cpp
    src/visualization/projector_4d.cpp
)

set(UI_SOURCES
    src/ui/application.cpp
    src/ui/controls.cpp
)

# Create simulation library
add_library(bec4d_simulation STATIC
    ${CORE_SOURCES}
    ${SIMULATION_SOURCES}
    ${SIMULATION_CUDA_SOURCES}
)

target_link_libraries(bec4d_simulation
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)

# Set CUDA properties
set_target_properties(bec4d_simulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Create visualization library
add_library(bec4d_visualization STATIC
    ${VISUALIZATION_SOURCES}
)

target_link_libraries(bec4d_visualization
    PUBLIC
        OpenGL::GL
        glfw
        glad::glad
        glm::glm
)

# Create main executable
add_executable(bec4d_sim
    src/main.cpp
    ${UI_SOURCES}
)

target_link_libraries(bec4d_sim
    PRIVATE
        bec4d_simulation
        bec4d_visualization
        imgui::imgui
)

# Optional: Link HDF5 if found
if(HDF5_FOUND)
    target_link_libraries(bec4d_simulation PUBLIC ${HDF5_CXX_LIBRARIES})
    target_include_directories(bec4d_simulation PUBLIC ${HDF5_INCLUDE_DIRS})
    target_compile_definitions(bec4d_simulation PUBLIC HAS_HDF5)
endif()

# Windows-specific settings
if(WIN32)
    target_compile_definitions(bec4d_sim PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Install targets
install(TARGETS bec4d_sim DESTINATION bin)
