#pragma once

#include "simulation_params.h"
#include "hypersphere_bec.h"
#include "renderer.h"
#include "projector_4d.h"
#include <memory>
#include <string>

namespace bec4d {

/**
 * Main application class
 * Manages simulation, visualization, and UI
 */
class Application {
public:
    Application();
    ~Application();

    // Main loop
    void run();

    // Configuration
    void setSimulationParams(const SimulationParams& params);
    void setWindowSize(int width, int height);

    // Simulation control
    void startSimulation(int n_steps, int save_every = 100);
    void pauseSimulation();
    void resumeSimulation();
    bool isSimulationRunning() const { return sim_running_; }

    // Data export/import
    void exportToHDF5(const std::string& filename);
    void loadFromHDF5(const std::string& filename);

private:
    void initUI();
    void updateUI();
    void processInput(float dt);
    void updateVisualization();
    void render();

    // Simulation state
    SimulationParams sim_params_;
    std::unique_ptr<HypersphereBEC> simulation_;
    bool sim_running_;
    int current_snapshot_;

    // Visualization state
    std::unique_ptr<Renderer> renderer_;
    std::unique_ptr<Projector4D> projector_;

    // UI state
    struct UIState {
        // Projection settings
        Projector4D::ProjectionMethod projection_method;
        float perspective_distance;

        // 4D rotation angles (viewing angles, not physics)
        float angle_xw, angle_yw, angle_zw;
        float angle_xy, angle_xz, angle_yz;

        // Display settings
        float point_size;
        float vortex_size;
        bool show_poles;
        bool show_axes;
        bool show_vortices;
        int render_subsample;  // Render 1 in N particles (further reduces already-subsampled snapshots)

        // Animation
        bool auto_rotate_4d;
        float rotation_speed;
        bool playing;
        float playback_speed;

        // Performance
        float fps;
        float sim_time_ms;
        int particles_rendered;
    } ui_state_;

    // Timing
    double last_frame_time_;
    double simulation_time_;
};

} // namespace bec4d
