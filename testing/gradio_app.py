import gradio as gr

# --- Dummy model functions (placeholders) ---
def run_inference(model_choice, file):
    return f"✅ Model '{model_choice}' processed file: {file.name if file else 'No file uploaded'}"

# --- UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Fall & Hazard Detection")

    # Dropdown for model selection
    model_choice = gr.Dropdown(
        choices=["Fall Detection (Pretrained)", "Hazard Detection (Pretrained)", "Fusion (Fall + Hazard)"],
        label="Select Model",
        value="Fall Detection (Pretrained)"
    )

    # Upload for image or video
    file_input = gr.File(label="Upload Image or Video", file_types=[".jpg", ".png", ".mp4", ".avi"])

    # Run button
    run_button = gr.Button("Run")

    # Output text for now (you can later make it image/video preview)
    output = gr.Textbox(label="Result")

    # Link logic
    run_button.click(fn=run_inference, inputs=[model_choice, file_input], outputs=output)

# --- Launch the app ---
if __name__ == "__main__":
    demo.launch(share=True)   # gets you a public link


