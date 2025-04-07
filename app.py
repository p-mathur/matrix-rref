import base64
import copy
import json
import os
from fractions import Fraction
from io import BytesIO

import numpy as np
import openai
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv


# === Setup ===

def load_openai_key():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


# === Utility ===

def format_fraction(value):
    frac = Fraction(value).limit_denominator()
    return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"


def describe_row_op(i, r, factor):
    factor_frac = format_fraction(factor)
    if factor_frac == "1":
        return f"R{i + 1} ← R{i + 1} - R{r + 1}"
    elif factor_frac == "-1":
        return f"R{i + 1} ← R{i + 1} + R{r + 1}"
    elif factor < 0:
        return f"R{i + 1} ← R{i + 1} + {format_fraction(abs(factor))} R{r + 1}"
    else:
        return f"R{i + 1} ← R{i + 1} - {factor_frac} R{r + 1}"


def highlight_pivot_positions(pivot_positions):
    def highlight(data):
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        for r, c in pivot_positions:
            if r in data.index and c in data.columns:
                styles.loc[r, c] = 'background-color: #ffff99'
        return styles

    return highlight


def image_to_data_url(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


# === LLM Extraction ===

def extract_matrix_from_image(image):
    try:
        data_url = image_to_data_url(image)
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {
                "role": "system",
                "content": (
                    "You are an expert at extracting numerical data from images of mathematical problems.\n"
                    "Your task is to return the numerical data as a matrix in JSON format: {\"matrix\": [[...]]}.\n"
                    "If the image shows a system of equations with both a matrix A and a vector b, combine them into a single augmented matrix [A | b].\n"
                    "If the image shows multiple vectors (e.g. x₁, x₂, x₃) or matrices side-by-side, stack them as columns of a single matrix.\n"
                    "Preserve ALL rows exactly as shown. For instance, if the image visually has 5 rows, your output must have 5 rows.\n"
                    "Do not omit, reorder, or guess fewer rows than appear.\n"
                    "Output only valid JSON, with no extra text or explanation.\n"
                )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Extract the matrix or augmented matrix from this image. Only respond with valid JSON."},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            max_tokens=500,
            temperature=0
        )
        reply_content = response['choices'][0]['message']['content'].strip()
        # print(reply_content)
        if not reply_content.startswith('{'):
            raise ValueError("LLM returned non-JSON content.")
        data = json.loads(reply_content)
        if "matrix" not in data:
            raise ValueError("Key 'matrix' not found.")
        return data["matrix"]
    except Exception as e:
        raise ValueError(f"LLM parsing failed: {str(e)}")


# === RREF Algorithm ===

def gaussian_elimination_steps(matrix):
    A = copy.deepcopy(matrix)
    steps = []
    pivot_positions = []
    rows, cols = len(A), len(A[0])
    r = 0

    for c in range(cols):
        pivot = next((i for i in range(r, rows) if abs(A[i][c]) > 1e-12), None)
        if pivot is None:
            continue

        pivot_positions.append((r, c))

        if pivot != r:
            A[r], A[pivot] = A[pivot], A[r]
            steps.append({"desc": f"Swap R{r + 1} ↔ R{pivot + 1}", "matrix": np.array(A).tolist()})

        pivot_val = A[r][c]
        scale_factor = 1 / pivot_val
        A[r] = [x / pivot_val for x in A[r]]
        if abs(scale_factor - 1.0) > 1e-12:
            steps.append({"desc": f"Scale R{r + 1} by {format_fraction(scale_factor)}", "matrix": np.array(A).tolist()})

        for i in range(rows):
            if i != r:
                factor = A[i][c]
                if abs(factor) < 1e-12:
                    continue
                A[i] = [A[i][j] - factor * A[r][j] for j in range(cols)]
                steps.append({
                    "desc": describe_row_op(i, r, factor),
                    "matrix": np.array(A).tolist()
                })

        r += 1
        if r == rows:
            break

    return steps, A, pivot_positions


# === UI Helpers ===

def display_uploaded_image(image):
    st.image(image, caption="Uploaded Image", use_container_width=True)


def render_matrix(matrix, pivot_positions):
    df = pd.DataFrame(matrix)
    styled = df.style \
        .format(format_fraction) \
        .apply(highlight_pivot_positions(pivot_positions), axis=None) \
        .hide(axis="index") \
        .hide(axis="columns")
    st.markdown(styled.to_html(), unsafe_allow_html=True)


def run_rref_workflow(matrix):
    steps, rref, pivot_positions = gaussian_elimination_steps(matrix)
    st.subheader("Step-by-Step RREF Process:")
    for step in steps:
        st.markdown(f"**Operation:** {step['desc']}")
        render_matrix(step["matrix"], pivot_positions)
    st.subheader("Final Reduced Row Echelon Form:")
    render_matrix(rref, pivot_positions)


# === Main App ===

def main():
    load_openai_key()

    st.title("Matrix RREF Converter with Vision + LLM")
    st.write("Upload an image of a matrix. The app uses OpenAI Vision to extract it and compute RREF.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        display_uploaded_image(image)

        try:
            matrix = extract_matrix_from_image(image)
            st.subheader("Parsed Matrix:")
            st.write(np.array(matrix))
        except Exception as e:
            st.error(f"Error parsing matrix: {e}")
            return

        if st.button("Convert to RREF"):
            run_rref_workflow(matrix)


if __name__ == "__main__":
    main()
