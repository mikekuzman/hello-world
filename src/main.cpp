#include "application.h"
#include "simulation_params.h"
#include <iostream>
#include <exception>

int main(int argc, char** argv) {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "  4D BEC Simulator - C++/CUDA Version  " << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;

        // Create application
        bec4d::Application app;

        // Configure simulation parameters
        bec4d::SimulationParams params;

        // Full scale parameters (dimensionless units: hbar=m=xi=1)
        params.R = 1000.0f;          // Hypersphere radius in healing lengths
        params.delta = 25.0f;        // Shell thickness in healing lengths
        params.g = 0.05f;            // Interaction strength (weak)
        params.omega = 0.03f;        // Rotation rate (moderate)
        params.N = 128;              // Grid points per dimension
        params.dt = 0.001f;          // Time step
        params.n_neighbors = 6;      // Number of neighbors for gradient calculation
        params.random_seed = 42;
        params.initial_condition_type = bec4d::SimulationParams::InitialConditionType::UniformNoise;

        params.updateDerived();
        params.print();

        app.setSimulationParams(params);
        app.setWindowSize(1920, 1080);

        // Run simulation BEFORE opening window
        std::cout << "\nRunning simulation (100 steps)..." << std::endl;
        app.startSimulation(100, 20);  // 100 steps, save every 20

        // Run application (will start simulation and visualization)
        std::cout << "\nStarting application..." << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  Mouse: Rotate camera" << std::endl;
        std::cout << "  Scroll: Zoom" << std::endl;
        std::cout << "  UI: Configure 4D rotation, projection, and visualization" << std::endl;
        std::cout << std::endl;

        app.run();

        std::cout << "\nApplication closed normally." << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "ERROR: Unknown exception" << std::endl;
        return 1;
    }
}
