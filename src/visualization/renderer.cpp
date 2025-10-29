#include "renderer.h"
#include "shader.h"
#include "camera.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

namespace bec4d {

// Vertex shaders will be embedded as strings
const char* particle_vertex_shader = R"(
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in float color;
layout(location = 2) in float brightness;

uniform mat4 viewProjection;
uniform float pointSize;

out float v_color;
out float v_brightness;

void main() {
    gl_Position = viewProjection * vec4(position, 1.0);
    gl_PointSize = pointSize;
    v_color = color;
    v_brightness = brightness;
}
)";

const char* particle_fragment_shader = R"(
#version 330 core
in float v_color;
in float v_brightness;

out vec4 fragColor;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // Phase angle to hue (color cycles through rainbow)
    vec3 color_rgb = hsv2rgb(vec3(v_color / 6.28318, 1.0, v_brightness));
    fragColor = vec4(color_rgb, 0.8);
}
)";

Renderer::Renderer(int width, int height)
    : window_(nullptr)
    , width_(width)
    , height_(height)
    , background_color_(0.05f, 0.05f, 0.1f)
    , point_size_(2.0f)
    , vortex_size_(0.05f)
    , show_poles_(true)
    , show_axes_(true)
    , last_mouse_x_(0.0)
    , last_mouse_y_(0.0)
    , mouse_pressed_(false)
{
    initGL();
    createParticleBuffers();
    createVortexBuffers();
    createPoleBuffers();
    createAxesBuffers();

    // Create shaders
    particle_shader_ = std::make_unique<Shader>(particle_vertex_shader, particle_fragment_shader);

    // TODO: Create other shaders (vortex, pole, axes)

    // Create camera
    camera_ = std::make_unique<Camera>(45.0f, static_cast<float>(width) / height, 0.1f, 1000.0f);
    camera_->setPosition(glm::vec3(0.0f, 0.0f, 5.0f));
    camera_->setTarget(glm::vec3(0.0f, 0.0f, 0.0f));
}

Renderer::~Renderer() {
    if (window_) {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

void Renderer::initGL() {
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA

    window_ = glfwCreateWindow(width_, height_, "4D BEC Simulator", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);  // VSync

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwDestroyWindow(window_);
        glfwTerminate();
        throw std::runtime_error("Failed to initialize GLAD");
    }

    // Enable features
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_MULTISAMPLE);

    std::cout << "OpenGL initialized" << std::endl;
    std::cout << "  Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "  Renderer: " << glGetString(GL_RENDERER) << std::endl;
}

void Renderer::createParticleBuffers() {
    glGenVertexArrays(1, &particle_buffers_.vao);
    glGenBuffers(1, &particle_buffers_.vbo_positions);
    glGenBuffers(1, &particle_buffers_.vbo_colors);
    glGenBuffers(1, &particle_buffers_.vbo_brightness);

    particle_buffers_.count = 0;

    // Setup VAO layout
    glBindVertexArray(particle_buffers_.vao);

    // Position (vec3)
    glBindBuffer(GL_ARRAY_BUFFER, particle_buffers_.vbo_positions);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Color (float - phase angle)
    glBindBuffer(GL_ARRAY_BUFFER, particle_buffers_.vbo_colors);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Brightness (float - density)
    glBindBuffer(GL_ARRAY_BUFFER, particle_buffers_.vbo_brightness);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, nullptr);

    glBindVertexArray(0);

    std::cout << "Created particle VAO: " << particle_buffers_.vao << std::endl;
}

void Renderer::createVortexBuffers() {
    // TODO: Implement
}

void Renderer::createPoleBuffers() {
    // TODO: Implement
}

void Renderer::createAxesBuffers() {
    // TODO: Implement
}

bool Renderer::shouldClose() const {
    return glfwWindowShouldClose(window_);
}

void Renderer::pollEvents() {
    glfwPollEvents();
}

void Renderer::swapBuffers() {
    glfwSwapBuffers(window_);
}

void Renderer::beginFrame() {
    glClearColor(background_color_.r, background_color_.g, background_color_.b, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::endFrame() {
    // Nothing special needed
}

void Renderer::uploadParticles(const ParticleData& data) {
    particle_buffers_.count = data.positions.size();

    if (particle_buffers_.count == 0) return;

    // Upload positions
    glBindBuffer(GL_ARRAY_BUFFER, particle_buffers_.vbo_positions);
    glBufferData(GL_ARRAY_BUFFER,
                 data.positions.size() * sizeof(glm::vec3),
                 data.positions.data(),
                 GL_DYNAMIC_DRAW);

    // Upload colors
    glBindBuffer(GL_ARRAY_BUFFER, particle_buffers_.vbo_colors);
    glBufferData(GL_ARRAY_BUFFER,
                 data.colors.size() * sizeof(float),
                 data.colors.data(),
                 GL_DYNAMIC_DRAW);

    // Upload brightness
    glBindBuffer(GL_ARRAY_BUFFER, particle_buffers_.vbo_brightness);
    glBufferData(GL_ARRAY_BUFFER,
                 data.brightness.size() * sizeof(float),
                 data.brightness.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Renderer::renderParticles(float point_size) {
    if (particle_buffers_.count == 0) return;

    particle_shader_->use();
    particle_shader_->setMat4("viewProjection", camera_->getViewProjectionMatrix());
    particle_shader_->setFloat("pointSize", point_size);

    glBindVertexArray(particle_buffers_.vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particle_buffers_.count));
    glBindVertexArray(0);
}

void Renderer::resize(int width, int height) {
    width_ = width;
    height_ = height;
    glViewport(0, 0, width, height);
    camera_->setAspectRatio(static_cast<float>(width) / height);
}

void Renderer::setBackgroundColor(const glm::vec3& color) {
    background_color_ = color;
}

void Renderer::uploadVortices(const std::vector<glm::vec3>& positions) {
    // TODO: Implement
}

void Renderer::uploadPoles(const glm::vec3& north, const glm::vec3& south) {
    // TODO: Implement
}

void Renderer::renderVortices(float vortex_size) {
    // TODO: Implement
}

void Renderer::renderPoles() {
    if (!show_poles_) return;
    // TODO: Implement
}

void Renderer::renderAxes() {
    if (!show_axes_) return;
    // TODO: Implement
}

void Renderer::updateCameraFromInput(float dt) {
    // TODO: Handle keyboard/mouse input for camera control
}

} // namespace bec4d
