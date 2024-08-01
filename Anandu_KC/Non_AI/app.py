import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

def display_image_streamlit(image, cmap='gray'):
    """Converts an image to a buffer and returns it."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap=cmap)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

def count_screws(image, min_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)  # Binary thresholding

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    screw_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            screw_count += 1
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(image, str(area), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return screw_count, image, thresh

def main():
    st.title('Non_AI Counting')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Read image file
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Threshold slider
        min_area = st.slider('Minimum area threshold', min_value=100, max_value=10000, value=5500)

        # Process image
        screw_count, result_image, binary_image = count_screws(image, min_area)

        # Display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Binary Thresholded Image')
            binary_image_buf = display_image_streamlit(binary_image)
            st.image(binary_image_buf, use_column_width=True)

        with col2:
            st.subheader('Detected with Contours')
            result_image_buf = display_image_streamlit(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            st.image(result_image_buf, use_column_width=True)

        st.write(f"Number of screws detected: {screw_count}")

if __name__ == "__main__":
    main()
