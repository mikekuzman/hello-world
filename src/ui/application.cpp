#include "application.h"
#include <GLFW/glfw3.h>
#include <iostream>

namespace bec4d {

Application::Application()
    : sim_running_(false)
    , current_snapshot_(0)
    , last_frame_time_(0.0)
    , simulation_time_(0.0)
{
    // Initialize default UI state
    ui_state_.projection_method = Projector4D::ProjectionMethod::Perspective;
    ui_state_.perspective_distance = 2.5f;
    ui_state_.angle_xw = 0.0f;
    ui_state_.angle_yw = 0.0f;
    ui_state_.angle_zw = 0.0f;
    ui_state_.angle_xy = 0.0f;
    ui_state_.angle_xz = 0.0f;
    ui_state_.angle_yz = 0.0f;
    ui_state_.point_size = 2.0f;
    ui_state_.vortex_size = 0.05f;
    ui_state_.show_poles = true;
    ui_state_.show_axes = true;
    ui_state_.show_vortices = true;
    ui_state_.auto_rotate_4d = false;
    ui_state_.rotation_speed = 0.1f;
    ui_state_.playing = false;
    ui_state_.playback_speed = 1.0f;
    ui_state_.fps = 0.0f;
    ui_state_.sim_time_ms = 0.0f;
    ui_state_.particles_rendered = 0;
}

Application::~Application() = default;

void Application::setSimulationParams(const SimulationParams& params) {
    sim_params_ = params;
}

void Application::setWindowSize(int width, int height) {
    renderer_ = std::make_unique<Renderer>(width, height);
}

void Application::run() {
    // Initialize renderer if not already done
    if (!renderer_) {
        renderer_ = std::make_unique<Renderer>(1920, 1080);
    }

    // Initialize projector
    projector_ = std::make_unique<Projector4D>(sim_params_.R, sim_params_.delta);
    projector_->setProjectionMethod(ui_state_.projection_method);
    projector_->setPerspectiveDistance(ui_state_.perspective_distance);

    initUI();

    std::cout << "\nApplication ready. Starting main loop..." << std::endl;

    // Main loop
    while (!renderer_->shouldClose()) {
        double current_time = glfwGetTime();
        float dt = static_cast<float>(current_time - last_frame_time_);
        last_frame_time_ = current_time;

        processInput(dt);
        updateUI();

        if (sim_running_) {
            // Run simulation step (if applicable)
            // For now, just visualize existing snapshots
        }

        updateVisualization();
        render();

        renderer_->pollEvents();
        renderer_->swapBuffers();

        // Update FPS
        ui_state_.fps = 1.0f / dt;
    }

    std::cout << "Application exiting..." << std::endl;
}

void Application::initUI() {
    // TODO: Initialize ImGui context
    std::cout << "UI initialized (stub)" << std::endl;
}

void Application::updateUI() {
    // TODO: Render ImGui panels
    // - Simulation controls
    // - Projection settings
    // - 4D rotation sliders
    // - Display options
    // - Performance metrics
}

void Application::processInput(float dt) {
    // TODO: Handle keyboard input
    renderer_->updateCameraFromInput(dt);

    // Auto-rotate in 4D if enabled
    if (ui_state_.auto_rotate_4d) {
        ui_state_.angle_xw += ui_state_.rotation_speed * dt;
        ui_state_.angle_yw += ui_state_.rotation_speed * dt * 0.7f;
        ui_state_.angle_zw += ui_state_.rotation_speed * dt * 0.5f;
    }
}

void Application::updateVisualization() {
    if (!simulation_ || simulation_->getSnapshots().empty()) {
        return;
    }

    const auto& snapshot = simulation_->getSnapshots()[current_snapshot_];

    // Convert flat coord array to vec4
    std::vector<glm::vec4> points_4d(snapshot.coords.size() / 4);
    for (size_t i = 0; i < points_4d.size(); ++i) {
        points_4d[i] = glm::vec4(
            snapshot.coords[i * 4 + 0],
            snapshot.coords[i * 4 + 1],
            snapshot.coords[i * 4 + 2],
            snapshot.coords[i * 4 + 3]
        );
    }

    // Apply 4D rotation
    projector_->rotatePoints4D(
        points_4d,
        ui_state_.angle_xy,
        ui_state_.angle_xz,
        ui_state_.angle_xw,
        ui_state_.angle_yz,
        ui_state_.angle_yw,
        ui_state_.angle_zw
    );

    // Project to 3D
    std::vector<glm::vec3> points_3d;
    projector_->projectPoints(points_4d, points_3d);

    // Upload to renderer
    Renderer::ParticleData particle_data;
    particle_data.positions = points_3d;
    particle_data.colors = snapshot.phase;
    particle_data.brightness = snapshot.density;

    renderer_->uploadParticles(particle_data);

    ui_state_.particles_rendered = static_cast<int>(points_3d.size());
}

void Application::render() {
    renderer_->beginFrame();

    // Render particles
    renderer_->renderParticles(ui_state_.point_size);

    // Render vortices
    if (ui_state_.show_vortices) {
        renderer_->renderVortices(ui_state_.vortex_size);
    }

    // Render poles
    if (ui_state_.show_poles) {
        renderer_->renderPoles();
    }

    // Render axes
    if (ui_state_.show_axes) {
        renderer_->renderAxes();
    }

    renderer_->endFrame();
}

void Application::startSimulation(int n_steps, int save_every) {
    std::cout << "Starting simulation..." << std::endl;

    simulation_ = std::make_unique<HypersphereBEC>(sim_params_);
    simulation_->run(n_steps, save_every);

    sim_running_ = false;  // Simulation completed

    std::cout << "Simulation complete. " << simulation_->getSnapshots().size()
              << " snapshots captured." << std::endl;
}

void Application::pauseSimulation() {
    sim_running_ = false;
}

void Application::resumeSimulation() {
    sim_running_ = true;
}

void Application::exportToHDF5(const std::string& filename) {
    std::cout << "Exporting to HDF5: " << filename << " (not implemented)" << std::endl;
    // TODO: Implement HDF5 export
}

void Application::loadFromHDF5(const std::string& filename) {
    std::cout << "Loading from HDF5: " << filename << " (not implemented)" << std::endl;
    // TODO: Implement HDF5 load
}

} // namespace bec4d
