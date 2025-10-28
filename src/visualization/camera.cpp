#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace bec4d {

Camera::Camera(float fov, float aspect, float near, float far)
    : position_(0.0f, 0.0f, 5.0f)
    , target_(0.0f, 0.0f, 0.0f)
    , up_(0.0f, 1.0f, 0.0f)
    , fov_(fov)
    , aspect_(aspect)
    , near_(near)
    , far_(far)
    , distance_(5.0f)
    , yaw_(0.0f)
    , pitch_(0.0f)
    , view_dirty_(true)
    , projection_dirty_(true)
{
}

glm::mat4 Camera::getViewMatrix() const {
    if (view_dirty_) {
        view_matrix_ = glm::lookAt(position_, target_, up_);
        view_dirty_ = false;
    }
    return view_matrix_;
}

glm::mat4 Camera::getProjectionMatrix() const {
    if (projection_dirty_) {
        projection_matrix_ = glm::perspective(glm::radians(fov_), aspect_, near_, far_);
        projection_dirty_ = false;
    }
    return projection_matrix_;
}

glm::mat4 Camera::getViewProjectionMatrix() const {
    return getProjectionMatrix() * getViewMatrix();
}

void Camera::rotate(float delta_yaw, float delta_pitch) {
    yaw_ += delta_yaw;
    pitch_ += delta_pitch;

    // Clamp pitch
    pitch_ = glm::clamp(pitch_, -89.0f, 89.0f);

    // Update position based on spherical coordinates
    updateViewMatrix();
}

void Camera::zoom(float delta) {
    distance_ -= delta;
    distance_ = glm::max(0.1f, distance_);
    updateViewMatrix();
}

void Camera::pan(float dx, float dy) {
    glm::vec3 right = glm::normalize(glm::cross(position_ - target_, up_));
    glm::vec3 up = glm::normalize(glm::cross(right, position_ - target_));

    target_ += right * dx + up * dy;
    position_ += right * dx + up * dy;

    view_dirty_ = true;
}

void Camera::updateViewMatrix() {
    // Convert spherical to Cartesian
    float yaw_rad = glm::radians(yaw_);
    float pitch_rad = glm::radians(pitch_);

    position_.x = target_.x + distance_ * cos(pitch_rad) * cos(yaw_rad);
    position_.y = target_.y + distance_ * sin(pitch_rad);
    position_.z = target_.z + distance_ * cos(pitch_rad) * sin(yaw_rad);

    view_dirty_ = true;
}

float Camera::getDistance() const {
    return glm::length(position_ - target_);
}

void Camera::setAspectRatio(float aspect) {
    aspect_ = aspect;
    projection_dirty_ = true;
}

} // namespace bec4d
