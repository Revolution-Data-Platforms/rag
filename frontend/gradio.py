import gradio as gr
import os
import time

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def bot(history):
    response = "**That's cool!**"
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=0.5, min_width=600):
            pass
        with gr.Column(scale=0.5, min_width=600):
            # create a drop down menu where every option is a function
            chat_history_dropdown = gr.Dropdown(
                choices=["Chat history will appear here."],
                label="Chat history",
                type="value",
            )

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png")),
    ))

    with gr.Row():
        with gr.Column(scale=0.8, min_width=800):
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
                container=False,
            )
        with gr.Column(scale=0.2, min_width=100):
            btn = gr.UploadButton("📁", file_types=["file", "image", "video", "audio"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
if __name__ == "__main__":
    demo.launch(allowed_paths=["avatar.png"])
