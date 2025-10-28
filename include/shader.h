#pragma once

#include <string>
#include <glm/glm.hpp>

namespace bec4d {

/**
 * OpenGL shader wrapper
 */
class Shader {
public:
    Shader(const std::string& vertex_source, const std::string& fragment_source);
    ~Shader();

    // Disable copy
    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    void use() const;
    uint32_t getProgram() const { return program_; }

    // Uniform setters
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec3(const std::string& name, const glm::vec3& value) const;
    void setVec4(const std::string& name, const glm::vec4& value) const;
    void setMat4(const std::string& name, const glm::mat4& value) const;

private:
    uint32_t compileShader(uint32_t type, const std::string& source);
    uint32_t linkProgram(uint32_t vertex, uint32_t fragment);

    uint32_t program_;
};

} // namespace bec4d
