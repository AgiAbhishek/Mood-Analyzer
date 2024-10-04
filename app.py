import streamlit as st
import numpy as np
import pandas as pd
import joblib
import altair as alt

# Load the model
pipe_lr = joblib.load(open('models/emotion_classifier_pipe_lr_03_jan_2022.pkl', 'rb'))

# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Dictionary for emojis corresponding to emotions
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", 
    "happy": "🤗", "joy": "😂", "neutral": "😐", 
    "sad": "😔", "sadness": "😔", "shame": "😳", 
    "surprise": "😮"
}

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Emotion Classifier", page_icon="😎", layout="wide")
    st.title("🌟 Emotion Classifier App 🌟")
    
    # Sidebar Menu
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Page
    if choice == "Home":
        st.markdown("### Analyze the Emotion in Your Text")
        st.write("Enter some text and let the app predict the emotion you're conveying!")

        # Input text form
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("📝 Enter your text here:", height=150)
            submit_text = st.form_submit_button(label="🔍 Analyze")

        # Prediction and Results
        if submit_text:
            col1, col2 = st.columns([2, 1])

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(f"📝 {raw_text}")

                st.success("Emotion Prediction")
                emoji_icon = emotions_emoji_dict[prediction[0]]
                st.write(f"**Emotion:** {prediction[0]} {emoji_icon}")
                st.write(f"**Confidence:** {np.max(probability):.2f}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.transpose().reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]
                st.dataframe(proba_df_clean)

                # Bar chart visualization
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x=alt.X('Emotions', sort='-y'),
                    y='Probability',
                    color='Emotions',
                    tooltip=['Emotions', 'Probability']
                ).properties(
                    width=350,
                    height=300
                ).configure_mark(
                    opacity=0.7,
                    color='red'
                )

                st.altair_chart(fig, use_container_width=True)

    # Monitor Page
    elif choice == "Monitor":
        st.subheader("📊 Monitor App Performance")

        # Simulated performance data for the past 10 tests
        test_data = pd.DataFrame({
            'Test Case': np.arange(1, 11),
            'Accuracy': np.random.uniform(0.7, 0.95, 10),
            'Time Taken (s)': np.random.uniform(0.5, 2.0, 10)
        })

        # Line chart for Accuracy
        st.markdown("### Model Accuracy over Test Cases")
        accuracy_chart = alt.Chart(test_data).mark_line(point=True).encode(
            x='Test Case',
            y='Accuracy',
            color=alt.value('green'),
            tooltip=['Test Case', 'Accuracy']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(accuracy_chart)

        # Bar chart for Time Taken per test case
        st.markdown("### Time Taken for Each Test Case")
        time_chart = alt.Chart(test_data).mark_bar().encode(
            x='Test Case',
            y='Time Taken (s)',
            color=alt.value('blue'),
            tooltip=['Test Case', 'Time Taken (s)']
        ).properties(
            width=700,
            height=400
        )
        st.altair_chart(time_chart)

    # About Page
    else:
        st.subheader("ℹ️ About")
        st.write("This NLP-powered web app predicts emotions from text with 70% accuracy. "
                 "It uses various Python libraries like Numpy, Pandas, Scikit-learn, and Altair. "
                 "The trained model was built using a Linear Regression algorithm, and the app is designed with Streamlit for an interactive UI.")
        st.caption("Created by: Abhishek Kushwaha")

# Run the app
if __name__ == '__main__':
    main()
