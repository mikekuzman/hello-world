#include "projector_4d.h"
#include "math_utils.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace bec4d {

Projector4D::Projector4D(float radius, float shell_thickness)
    : radius_(radius)
    , shell_thickness_(shell_thickness)
    , perspective_distance_(2.5f)
    , method_(ProjectionMethod::Perspective)
{
}

glm::mat4 Projector4D::rotationMatrix4D(
    float angle_xy,
    float angle_xz,
    float angle_xw,
    float angle_yz,
    float angle_yw,
    float angle_zw
) const {
    // XY plane rotation (affects x and y, z and w unchanged)
    glm::mat4 R_xy = glm::mat4(1.0f);
    if (angle_xy != 0.0f) {
        float c = std::cos(angle_xy);
        float s = std::sin(angle_xy);
        R_xy[0][0] = c; R_xy[0][1] = -s;
        R_xy[1][0] = s; R_xy[1][1] = c;
    }

    // XZ plane rotation
    glm::mat4 R_xz = glm::mat4(1.0f);
    if (angle_xz != 0.0f) {
        float c = std::cos(angle_xz);
        float s = std::sin(angle_xz);
        R_xz[0][0] = c;  R_xz[0][2] = -s;
        R_xz[2][0] = s;  R_xz[2][2] = c;
    }

    // XW plane rotation (involves 4th dimension!)
    glm::mat4 R_xw = glm::mat4(1.0f);
    if (angle_xw != 0.0f) {
        float c = std::cos(angle_xw);
        float s = std::sin(angle_xw);
        R_xw[0][0] = c;  R_xw[0][3] = -s;
        R_xw[3][0] = s;  R_xw[3][3] = c;
    }

    // YZ plane rotation
    glm::mat4 R_yz = glm::mat4(1.0f);
    if (angle_yz != 0.0f) {
        float c = std::cos(angle_yz);
        float s = std::sin(angle_yz);
        R_yz[1][1] = c;  R_yz[1][2] = -s;
        R_yz[2][1] = s;  R_yz[2][2] = c;
    }

    // YW plane rotation (involves 4th dimension!)
    glm::mat4 R_yw = glm::mat4(1.0f);
    if (angle_yw != 0.0f) {
        float c = std::cos(angle_yw);
        float s = std::sin(angle_yw);
        R_yw[1][1] = c;  R_yw[1][3] = -s;
        R_yw[3][1] = s;  R_yw[3][3] = c;
    }

    // ZW plane rotation (involves 4th dimension!)
    glm::mat4 R_zw = glm::mat4(1.0f);
    if (angle_zw != 0.0f) {
        float c = std::cos(angle_zw);
        float s = std::sin(angle_zw);
        R_zw[2][2] = c;  R_zw[2][3] = -s;
        R_zw[3][2] = s;  R_zw[3][3] = c;
    }

    // Compose all rotations
    return R_xy * R_xz * R_xw * R_yz * R_yw * R_zw;
}

void Projector4D::rotatePoints4D(
    std::vector<glm::vec4>& points,
    float angle_xy,
    float angle_xz,
    float angle_xw,
    float angle_yz,
    float angle_yw,
    float angle_zw
) const {
    glm::mat4 R = rotationMatrix4D(angle_xy, angle_xz, angle_xw, angle_yz, angle_yw, angle_zw);

    for (auto& p : points) {
        p = R * p;
    }
}

glm::vec3 Projector4D::projectPoint(const glm::vec4& point_4d) const {
    switch (method_) {
        case ProjectionMethod::Perspective:
            return projectPerspective(point_4d);
        case ProjectionMethod::Stereographic:
            return projectStereographic(point_4d);
        case ProjectionMethod::Orthogonal:
            return projectOrthogonal(point_4d);
        default:
            return projectPerspective(point_4d);
    }
}

void Projector4D::projectPoints(
    const std::vector<glm::vec4>& points_4d,
    std::vector<glm::vec3>& points_3d_out
) const {
    points_3d_out.resize(points_4d.size());

    for (size_t i = 0; i < points_4d.size(); ++i) {
        points_3d_out[i] = projectPoint(points_4d[i]);
    }
}

void Projector4D::projectPointsRaw(
    const float* points_4d,
    size_t n_points,
    float* points_3d_out
) const {
    for (size_t i = 0; i < n_points; ++i) {
        glm::vec4 p4(points_4d[i * 4 + 0], points_4d[i * 4 + 1],
                     points_4d[i * 4 + 2], points_4d[i * 4 + 3]);
        glm::vec3 p3 = projectPoint(p4);

        points_3d_out[i * 3 + 0] = p3.x;
        points_3d_out[i * 3 + 1] = p3.y;
        points_3d_out[i * 3 + 2] = p3.z;
    }
}

glm::vec3 Projector4D::projectPerspective(const glm::vec4& p) const {
    // Perspective projection from viewpoint at (0, 0, 0, distance)
    // onto w=0 hyperplane
    float scale = perspective_distance_ / (perspective_distance_ - p.w);
    return glm::vec3(p.x * scale, p.y * scale, p.z * scale);
}

glm::vec3 Projector4D::projectStereographic(const glm::vec4& p) const {
    // Stereographic projection from north pole
    float scale = radius_ / (radius_ - p.w + 1e-10f);
    return glm::vec3(p.x * scale, p.y * scale, p.z * scale);
}

glm::vec3 Projector4D::projectOrthogonal(const glm::vec4& p) const {
    // Simply drop the w coordinate
    return glm::vec3(p.x, p.y, p.z);
}

std::vector<glm::vec4> Projector4D::generateRandomPointsOn4DSphere(size_t n_points) const {
    std::vector<glm::vec4> points;
    points.reserve(n_points);

    math::Random rng(12345);

    for (size_t i = 0; i < n_points; ++i) {
        // Generate from 4D Gaussian and normalize
        glm::vec4 p(rng.normal(), rng.normal(), rng.normal(), rng.normal());
        p = glm::normalize(p);

        // Add random radial variation for shell thickness
        float r = radius_ + (rng.uniform() - 0.5f) * shell_thickness_;
        p *= r;

        points.push_back(p);
    }

    return points;
}

} // namespace bec4d
