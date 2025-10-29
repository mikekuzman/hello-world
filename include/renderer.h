#pragma once

#include "projector_4d.h"
#include <memory>
#include <vector>
#include <glm/glm.hpp>

struct GLFWwindow;

namespace bec4d {

class Shader;
class Camera;

// Forward declare callbacks
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

/**
 * OpenGL renderer for 4D BEC visualization
 * Displays superfluid particles, vortices, and 4D pole markers
 */
class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();

    // Disable copy
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Friend callbacks
    friend void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    friend void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    friend void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    // Window management
    bool shouldClose() const;
    void pollEvents();
    void swapBuffers();
    GLFWwindow* getWindow() { return window_; }

    // Rendering
    void beginFrame();
    void endFrame();

    // Data upload
    struct ParticleData {
        std::vector<glm::vec3> positions;   // 3D projected positions
        std::vector<float> colors;          // Phase angle or density (1 float per particle)
        std::vector<float> brightness;      // Density (for alpha/size)
    };

    void uploadParticles(const ParticleData& data);
    void uploadVortices(const std::vector<glm::vec3>& positions);
    void uploadPoles(const glm::vec3& north, const glm::vec3& south);

    // Render calls
    void renderParticles(float point_size = 2.0f);
    void renderVortices(float vortex_size = 0.05f);
    void renderPoles();
    void renderAxes();  // 3D coordinate axes for reference

    // Camera control
    Camera& getCamera() { return *camera_; }
    void updateCameraFromInput(float dt);

    // Settings
    void setBackgroundColor(const glm::vec3& color);
    void setPointSize(float size) { point_size_ = size; }
    void setVortexSize(float size) { vortex_size_ = size; }
    void setShowPoles(bool show) { show_poles_ = show; }
    void setShowAxes(bool show) { show_axes_ = show; }

    // Viewport
    void resize(int width, int height);
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }

private:
    void initGL();
    void createParticleBuffers();
    void createVortexBuffers();
    void createPoleBuffers();
    void createAxesBuffers();

    // Window
    GLFWwindow* window_;
    int width_, height_;

    // OpenGL objects
    struct ParticleBuffers {
        uint32_t vao;
        uint32_t vbo_positions;
        uint32_t vbo_colors;
        uint32_t vbo_brightness;
        size_t count;
    } particle_buffers_;

    struct VortexBuffers {
        uint32_t vao;
        uint32_t vbo_positions;
        uint32_t ibo;
        size_t count;
        size_t index_count;
    } vortex_buffers_;

    struct PoleBuffers {
        uint32_t vao;
        uint32_t vbo_positions;
        uint32_t vbo_colors;
    } pole_buffers_;

    struct AxesBuffers {
        uint32_t vao;
        uint32_t vbo;
        uint32_t ibo;
    } axes_buffers_;

    // Shaders
    std::unique_ptr<Shader> particle_shader_;
    std::unique_ptr<Shader> vortex_shader_;
    std::unique_ptr<Shader> pole_shader_;
    std::unique_ptr<Shader> axes_shader_;

    // Camera
    std::unique_ptr<Camera> camera_;

    // Settings
    glm::vec3 background_color_;
    float point_size_;
    float vortex_size_;
    bool show_poles_;
    bool show_axes_;

    // Mouse state for camera control
    double last_mouse_x_, last_mouse_y_;
    bool mouse_pressed_;
};

} // namespace bec4d
