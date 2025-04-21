import streamlit as st
import pickle
import pandas as pd
import os

# Load the model
model_loaded = pickle.load(open('diabetes_prediction_model', 'rb'))

def main():
    st.title("🩺 Early Stage Diabetes Risk Prediction")
    st.markdown("---")

    # Slideshow Section
    st.markdown("### 🖼️ Slideshow: Diabetes Awareness")
    image_folder = "images"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))])

    if image_files:
        selected_idx = st.slider("Slide", 1, len(image_files), 1)
        image_path = os.path.join(image_folder, image_files[selected_idx - 1])
        st.image(image_path, use_column_width=True, caption=f"Slide {selected_idx}")
    else:
        st.warning("No images found in the 'images' folder.")

    st.markdown("---")
    st.header("🧾 Fill in your details")

    with st.form(key='diabetes_form'):
        age = st.number_input("Enter Age", min_value=1.0, step=1.0)
        gender = st.selectbox("Gender", ["M", "F"])

        polyuria = st.selectbox("Do you have Polyuria?", ["yes", "no"])
        polydipsia = st.selectbox("Do you have Polydipsia?", ["yes", "no"])
        sudden_weight_loss = st.selectbox("Experiencing Sudden Weight Loss?", ["yes", "no"])
        weakness = st.selectbox("Do you feel Weakness?", ["yes", "no"])
        polyphagia = st.selectbox("Do you have Polyphagia?", ["yes", "no"])
        genital_thrush = st.selectbox("Do you have Genital Thrush?", ["yes", "no"])
        visual_blurring = st.selectbox("Experiencing Visual Blurring?", ["yes", "no"])
        irritability = st.selectbox("Are you Irritable?", ["yes", "no"])
        partial_paresis = st.selectbox("Do you have Partial Paresis?", ["yes", "no"])
        muscle_stiffness = st.selectbox("Experiencing Muscle Stiffness?", ["yes", "no"])
        alopecia = st.selectbox("Do you have Alopecia?", ["yes", "no"])

        submit = st.form_submit_button("Predict")

    if submit:
        data = {
            'Age': [age],
            'Gender': [gender],
            'Polyuria': [polyuria],
            'Polydipsia': [polydipsia],
            'sudden weight loss': [sudden_weight_loss],
            'weakness': [weakness],
            'Polyphagia': [polyphagia],
            'Genital thrush': [genital_thrush],
            'visual blurring': [visual_blurring],
            'Irritability': [irritability],
            'partial paresis': [partial_paresis],
            'muscle stiffness': [muscle_stiffness],
            'Alopecia': [alopecia]
        }

        df = pd.DataFrame(data)
        df.replace({"no": 0, "yes": 1}, inplace=True)
        df.replace({"M": 0, "F": 1}, inplace=True)

        symptoms = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                    'Genital thrush', 'visual blurring', 'Irritability', 'partial paresis',
                    'muscle stiffness', 'Alopecia']
        df['total_symptoms'] = df[symptoms].sum(axis=1)

        age_bins = [0, 20, 40, 60, 100]
        age_labels = ['1-20', '21-40', '41-60', '61-100']
        df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
        df['Age_Group_Binary'] = df['Age_Group'].apply(lambda x: 1 if x in ['41-60', '61-100'] else 0)
        df.drop(columns=['Age_Group'], inplace=True)

        prediction = model_loaded.predict(df)[0]
        result = '🟢 **Negative** - No early diabetes signs' if prediction == 0 else '🔴 **Positive** - Risk of early diabetes detected'

        st.success(f"The prediction result is: {result}")

if __name__ == "__main__":
    main()
