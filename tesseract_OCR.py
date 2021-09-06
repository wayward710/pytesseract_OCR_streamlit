import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import re
from gtts import gTTS
from utils import getSkewAngle, rotateImage, deskew


def streamlit_OCR():
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "Homepage",
            "Detection",
            "Boxes and Text",
            "OCR Numerical",
            "Text To Audio",
            "Align",
        ],
    )

    if page == "Homepage":
        st.title("OCR")
        # To display Images
        image_1 = Image.open("media/TheTesseract.jpg")
        image_2 = Image.open("media/Tesseract-OCR-Architecture-3.png")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Tesseract")
            st.image(image_1, caption="Tesseract")
        with col2:
            st.header("Tesseract Arcitecture")
            st.image(image_2, caption="Tesseract Arcitecture")
    elif page == "Detection":
        st.title("OCR Text detection with Tesseract")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            # uploading image
            image = Image.open(uploaded_file)
            # Converting image into RGB
            new_img = np.array(image.convert("RGB"))
            hImg, wImg, _ = new_img.shape
            img = cv2.cvtColor(new_img, 1)
            # Converting Image into Gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detecting data from image
            d = pytesseract.image_to_data(img, output_type=Output.DICT)
            n_boxes = len(d["text"])
            # Detecting only Text
            text = pytesseract.image_to_string(img)
            for i in range(n_boxes):
                if int(d["conf"][i]) > 60:
                    (x, y, w, h) = (
                        d["left"][i],
                        d["top"][i],
                        d["width"][i],
                        d["height"][i],
                    )
                    # Draw Rectangle boxes and Text around Text
                    cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        new_img,
                        d["text"][i],
                        (x, y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (255, 0, 0),
                        1,
                    )
            # Writing detected text
            st.write(text)
            # Printing image with boxes and text
            st.image(new_img)
    elif page == "Boxes and Text":
        st.title("OCR Boxes and Text around Text")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            new_img = np.array(image.convert("RGB"))
            hImg, wImg, _ = new_img.shape
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = pytesseract.image_to_data(img)
            for x, i in enumerate(boxes.splitlines()):
                if x != 0:
                    b = i.split()
                    if len(b) == 12:
                        (x, y, w, h) = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                        cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            new_img,
                            b[11],
                            (x, y),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (255, 0, 0),
                            2,
                        )
            st.image(img)
            st.image(new_img)
            st.write("Detecting...")
    elif page == "OCR Numerical":
        st.title("OCR Numerical detection with Tesseract (Custom)")

        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            new_img = np.array(image.convert("RGB"))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            d = pytesseract.image_to_data(image, output_type=Output.DICT)
            n_boxes = len(d["text"])
            for i in range(n_boxes):
                if int(d["conf"][i]) > 60:
                    if re.match(
                        r"^\d*[.,]\d*|\$\d*[.,]\d*|\d{2,5} |\$\d*$", d["text"][i]
                    ):
                        (x, y, w, h) = (
                            d["left"][i],
                            d["top"][i],
                            d["width"][i],
                            d["height"][i],
                        )
                        cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(
                            new_img,
                            d["text"][i],
                            (x, y),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
            st.image(new_img)
            st.write("")
            st.write("Detecting...")
    elif page == "Text To Audio":
        st.title("OCR Text To Audio")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            new_img = np.array(image.convert("RGB"))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(img)
            st.image(img)
            st.write(text)
            sound = gTTS(text, lang="en")
            sound.save("media/sound.mp3")
            audio_file = open("media/sound.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

    elif page == "Align":
        st.title("OCR Alignment")
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            new_img = np.array(image.convert("RGB"))
            img = cv2.cvtColor(new_img, 1)
            st.image(img, caption="Original Image")

            rotated_img = deskew(img)
            st.image(rotated_img, caption="Rotational Image")


if __name__ == "__main__":
    streamlit_OCR()
