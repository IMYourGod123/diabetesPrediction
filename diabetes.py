import streamlit as st
import pickle
import pandas as pd
import os
from streamlit_autorefresh import st_autorefresh

# Load the model
model_loaded = pickle.load(open('diabetes_prediction_model', 'rb'))

def main():
    st.set_page_config(page_title="Diabetes Risk Prediction", layout="centered")
    st.title("🩺 Early Stage Diabetes Risk Prediction | 糖尿病早期风险预测")
    st.markdown("---")

    # 🖼️ Slideshow Section
    st.markdown("### Diabetes Awareness")
    image_folder = "images"

    if os.path.exists(image_folder):
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))])
        
        if image_files:
            if "slide_index" not in st.session_state:
                st.session_state.slide_index = 0

            image_path = os.path.join(image_folder, image_files[st.session_state.slide_index])
            st.image(image_path, use_container_width=True, caption=f"Slide {st.session_state.slide_index + 1} | 幻灯片 {st.session_state.slide_index + 1}")

            # Auto refresh every 3 seconds
            st_autorefresh(interval=3000, key="slideshow_refresh")
            st.session_state.slide_index = (st.session_state.slide_index + 1) % len(image_files)
        else:
            st.warning("🖼️ No images in 'images' folder. ")
    else:
        st.warning("⚠️ 'images' folder not found. Please create it. ")

    st.markdown("---")
    st.header("🧾 Fill in your details | 填写您的资料")

    with st.form(key='diabetes_form'):
        age = st.number_input("Enter Age | 输入年龄", min_value=1.0, step=1.0)
        gender = st.selectbox("Gender | 性别", ["M (Male 男)", "F (Female 女)"])

        polyuria = st.selectbox("Do you have Polyuria? | 您是否有尿多？", ["yes 是", "no 否"])
        polydipsia = st.selectbox("Do you have Polydipsia? | 您是否有口渴？", ["yes 是", "no 否"])
        sudden_weight_loss = st.selectbox("Sudden Weight Loss? | 突然体重下降？", ["yes 是", "no 否"])
        weakness = st.selectbox("Do you feel Weakness? | 是否感到虚弱？", ["yes 是", "no 否"])
        polyphagia = st.selectbox("Do you have Polyphagia? | 您是否有多食？", ["yes 是", "no 否"])
        genital_thrush = st.selectbox("Genital Thrush? | 生殖器念珠菌感染？", ["yes 是", "no 否"])
        visual_blurring = st.selectbox("Visual Blurring? | 视力模糊？", ["yes 是", "no 否"])
        irritability = st.selectbox("Irritability? | 易怒？", ["yes 是", "no 否"])
        partial_paresis = st.selectbox("Partial Paresis? | 部分瘫痪？", ["yes 是", "no 否"])
        muscle_stiffness = st.selectbox("Muscle Stiffness? | 肌肉僵硬？", ["yes 是", "no 否"])
        alopecia = st.selectbox("Alopecia? | 脱发？", ["yes 是", "no 否"])

        submit = st.form_submit_button("Predict | 预测")

    if submit:
        # Mapping values
        def map_value(val):
            return 1 if "yes" in val else 0

        gender_val = 0 if "M" in gender else 1

        data = {
            'Age': [age],
            'Gender': [gender_val],
            'Polyuria': [map_value(polyuria)],
            'Polydipsia': [map_value(polydipsia)],
            'sudden weight loss': [map_value(sudden_weight_loss)],
            'weakness': [map_value(weakness)],
            'Polyphagia': [map_value(polyphagia)],
            'Genital thrush': [map_value(genital_thrush)],
            'visual blurring': [map_value(visual_blurring)],
            'Irritability': [map_value(irritability)],
            'partial paresis': [map_value(partial_paresis)],
            'muscle stiffness': [map_value(muscle_stiffness)],
            'Alopecia': [map_value(alopecia)]
        }

        df = pd.DataFrame(data)
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
        if prediction == 0:
            st.success("🟢 **Negative** - No early diabetes signs | 未发现糖尿病早期迹象")
        else:
            st.error("🔴 **Positive** - Risk of early diabetes detected | 检测到糖尿病早期风险")

if __name__ == "__main__":
    main()
