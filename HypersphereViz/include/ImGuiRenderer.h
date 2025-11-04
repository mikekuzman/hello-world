#pragma once

// ImGui stub - for future UI integration
// For now, we'll use keyboard controls only

class ImGuiRenderer
{
public:
    ImGuiRenderer() {}
    ~ImGuiRenderer() {}

    bool Initialize() { return true; }
    void Shutdown() {}
    void NewFrame() {}
    void Render() {}
};
