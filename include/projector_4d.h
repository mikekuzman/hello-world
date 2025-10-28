#pragma once

#include <vector>
#include <glm/glm.hpp>

namespace bec4d {

/**
 * 4D to 3D projection utilities
 * Implements perspective, stereographic, and orthogonal projections
 *
 * Matches Python HypersphereProjector class
 */
class Projector4D {
public:
    enum class ProjectionMethod {
        Perspective,
        Stereographic,
        Orthogonal
    };

    Projector4D(float radius = 1.0f, float shell_thickness = 0.01f);

    // Set projection parameters
    void setRadius(float r) { radius_ = r; }
    void setShellThickness(float t) { shell_thickness_ = t; }
    void setPerspectiveDistance(float d) { perspective_distance_ = d; }
    void setProjectionMethod(ProjectionMethod m) { method_ = m; }

    // 4D rotation matrix generation
    // In 4D there are 6 planes of rotation (vs 3 axes in 3D):
    // XY, XZ, XW, YZ, YW, ZW
    glm::mat4 rotationMatrix4D(
        float angle_xy = 0.0f,
        float angle_xz = 0.0f,
        float angle_xw = 0.0f,
        float angle_yz = 0.0f,
        float angle_yw = 0.0f,
        float angle_zw = 0.0f
    ) const;

    // Rotate 4D points
    void rotatePoints4D(
        std::vector<glm::vec4>& points,
        float angle_xy = 0.0f,
        float angle_xz = 0.0f,
        float angle_xw = 0.0f,
        float angle_yz = 0.0f,
        float angle_yw = 0.0f,
        float angle_zw = 0.0f
    ) const;

    // Project single point
    glm::vec3 projectPoint(const glm::vec4& point_4d) const;

    // Project array of points (batch operation)
    void projectPoints(
        const std::vector<glm::vec4>& points_4d,
        std::vector<glm::vec3>& points_3d_out
    ) const;

    // Projection from raw float array (interleaved: w,x,y,z,w,x,y,z,...)
    void projectPointsRaw(
        const float* points_4d,
        size_t n_points,
        float* points_3d_out
    ) const;

    // Generate test data (random points on 4D sphere)
    std::vector<glm::vec4> generateRandomPointsOn4DSphere(size_t n_points) const;

private:
    // Projection implementations
    glm::vec3 projectPerspective(const glm::vec4& p) const;
    glm::vec3 projectStereographic(const glm::vec4& p) const;
    glm::vec3 projectOrthogonal(const glm::vec4& p) const;

    // Parameters
    float radius_;
    float shell_thickness_;
    float perspective_distance_;
    ProjectionMethod method_;
};

} // namespace bec4d
