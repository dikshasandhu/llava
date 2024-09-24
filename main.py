import streamlit as st
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image

# Set up the LLaVA model for CPU
model_id = "llava-hf/llava-1.5-7b-hf"

# Load model and processor for CPU
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="cpu")

# Streamlit app UI
st.title("LLaVA: Image Captioning App (CPU Version)")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Prompt input
    prompt = st.text_area("Enter your prompt", "USER: <image>\nDescribe this image\nASSISTANT:")

    # Button to generate caption
    if st.button('Generate Caption'):
        with st.spinner('Generating caption...'):
            # Prepare image and prompt for CPU
            inputs = processor(prompt, images=[img], padding=True, return_tensors="pt").to("cpu")
            output = model.generate(**inputs, max_new_tokens=50)
            generated_text = processor.batch_decode(output, skip_special_tokens=True)
            for text in generated_text:
                st.write(f"Caption: {text.split('ASSISTANT:')[-1]}")
