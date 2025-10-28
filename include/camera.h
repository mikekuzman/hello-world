#pragma once

#include <glm/glm.hpp>

namespace bec4d {

/**
 * 3D camera for viewing the projected visualization
 */
class Camera {
public:
    Camera(float fov = 45.0f, float aspect = 16.0f/9.0f, float near = 0.1f, float far = 1000.0f);

    // View matrix
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::mat4 getViewProjectionMatrix() const;

    // Camera control
    void setPosition(const glm::vec3& pos) { position_ = pos; }
    void setTarget(const glm::vec3& target) { target_ = target; }
    void setUp(const glm::vec3& up) { up_ = up; }

    // Mouse-based rotation (arcball)
    void rotate(float delta_yaw, float delta_pitch);
    void zoom(float delta);
    void pan(float dx, float dy);

    // Getters
    const glm::vec3& getPosition() const { return position_; }
    const glm::vec3& getTarget() const { return target_; }
    float getDistance() const;

    // Update aspect ratio (on window resize)
    void setAspectRatio(float aspect);

private:
    void updateViewMatrix();

    glm::vec3 position_;
    glm::vec3 target_;
    glm::vec3 up_;

    float fov_;
    float aspect_;
    float near_;
    float far_;

    // Spherical coordinates (for arcball)
    float distance_;
    float yaw_;
    float pitch_;

    mutable glm::mat4 view_matrix_;
    mutable glm::mat4 projection_matrix_;
    mutable bool view_dirty_;
    mutable bool projection_dirty_;
};

} // namespace bec4d
