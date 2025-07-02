# Product Requirements Document (PRD): AI Watermark Remover v1.0

**1. Vision & Goal**
To create a comprehensive, user-friendly, and powerful AI tool for removing watermarks from images. The primary goal of this refactoring is to establish a clean, modular, and extensible architecture that simplifies maintenance and accelerates future development, such as integrating new AI models and features.

**2. Core User Scenarios**
*   **As a content creator,** I want to easily upload an image, have the watermark automatically detected, and get a high-quality, clean version of the image with minimal artifacts.
*   **As a power user,** I want to fine-tune every step of the process—from mask detection thresholds and dilation to inpainting model parameters—to achieve perfect results on challenging images.
*   **As a developer,** I want the codebase to be so well-structured that I can easily add a new inpainting model (like Stable Diffusion) or a new detection method with minimal effort.

**3. Key Features & Functionality**

| Category | Feature | Requirement |
| :--- | :--- | :--- |
| **Frontend (UI)** | Web Interface | A Streamlit-based GUI for all user interactions. |
| | Image Upload | Support for JPG, PNG, WEBP formats. |
| | Parameter Control | A detailed panel to adjust all backend parameters in real-time. |
| | Result Preview | An interactive before-and-after slider for immediate comparison. |
| | Image Download | Ability to download the result in PNG, WEBP, or JPG. |
| **Backend (Processing)** | Mask Generation | **1. Custom Model:** Use a dedicated segmentation model. <br> **2. Florence-2:** Use text-prompt-based detection. <br> **3. Manual Upload:** Allow user-provided masks. |
| | Mask Refinement | Control over mask threshold, dilation kernel size, and iterations. |
| | Inpainting | **1. LaMA Model:** High-quality, context-aware filling. <br> **2. Transparency Mode:** Option to output a transparent mask instead of filling. |
| | Configuration | All key paths, model names, and default parameters managed via `web_config.yaml`. |

**4. Future Roadmap Integration**
The new architecture must be designed to seamlessly accommodate the planned features from `NEXT_STEPS.md`:
*   **Model Extensibility:** Easily add new inpainting models (e.g., SD 1.5) and detection models.
*   **Batch Processing:** The backend logic should be callable in a loop for batch operations.
*   **Interactive Features:** The structure should support future integration of a manual mask editor.

**5. Technical Stack**
*   **Primary Language:** Python
*   **AI/ML:** PyTorch, Transformers, iopaint
*   **Web:** Streamlit
*   **Image Libs:** OpenCV, Pillow
