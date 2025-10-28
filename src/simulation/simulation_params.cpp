#include "simulation_params.h"
#include <iostream>
#include <iomanip>

namespace bec4d {

SimulationParams::SimulationParams() {
    updateDerived();
}

void SimulationParams::updateDerived() {
    box_size = R * 1.2f;
    dx = 2.0f * box_size / static_cast<float>(N);
}

void SimulationParams::print() const {
    std::cout << "======================================================================" << std::endl;
    std::cout << "4D BEC Simulator - High Performance C++/CUDA" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "Physical parameters:" << std::endl;
    std::cout << "  R = " << R << " ξ" << std::endl;
    std::cout << "  δ = " << delta << " ξ (R/δ = " << std::fixed << std::setprecision(1) << (R/delta) << ")" << std::endl;
    std::cout << "  g = " << g << ", Ω = " << omega << std::endl;
    std::cout << std::endl;
    std::cout << "Computational parameters:" << std::endl;
    std::cout << "  Grid: " << N << "^4" << std::endl;
    std::cout << "  dx = " << std::fixed << std::setprecision(3) << dx << " ξ" << std::endl;
    std::cout << "  dt = " << dt << std::endl;
    std::cout << "  Neighbors: " << n_neighbors << std::endl;
    std::cout << "  Random seed: " << random_seed << std::endl;
    std::cout << "======================================================================" << std::endl;
}

} // namespace bec4d
