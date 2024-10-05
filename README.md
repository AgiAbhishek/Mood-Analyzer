# MoodTracker
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/AgiAbhishek/Mood-Analyzer)](https://github.com/AgiAbhishek/Mood-Analyzer/stargazers)

## MoodTracker Live Demo

**Note:** Scroll down to access the projects

**MoodTracker** is a multi-modal mood analysis application that provides users with different ways to track and analyze their daily mood. This project combines three components: a real-time face emotion detection application, an emotion classifier app, and a voice-based mood analyzer. Each component offers unique features for mood tracking and analysis.

## Features

### [Emotion Classifier App (Text-Based Mood Analyzer)]
- An NLP-powered web app that can predict emotions from text recognition with 70 percent accuracy.
- Utilizes Python libraries including Numpy, Pandas, Seaborn, Scikit-learn, Scipy, Joblib, eli5, lime, neattext, altair, and Streamlit.
- Employs a Linear regression model from the scikit-learn library to train a dataset containing speeches and their respective emotions.
- Joblib is used for storing and using the trained model in the website.

## Requirements
The **MoodTracker** project requires the following dependencies for each component:

### Emotion Classifier App (Text-Based Mood Analyzer):
- Numpy
- Pandas
- Seaborn
- Scikit-learn
- Scipy
- Joblib
- eli5
- lime
- neattext
- altair
- Streamlit

## Usage
To use the **MoodTracker** application, follow the specific installation and execution instructions for each component.


2. **Emotion Classifier App (Text-Based Mood Analyzer)**
   - Install the required dependencies.
   - Navigate to the `NLP-Text-Emotion` folder:
     ```bash
     cd NLP-Text-Emotion
     ```
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the application:
     ```bash
     streamlit run app.py
     ```
   - Access the app via your web browser to enter text and analyze emotions.


## Note
Each component offers a different way to track and analyze your mood. Make sure to install the required dependencies for the component you wish to use.

## Combined Features
- **Mood Tracking**: Tracks daily mood using real-time face, text, and voice inputs.
- **Sentiment Analysis**: Performs sentiment analysis to determine mood.
- **Data Visualization**: Provides visual mood feedback using Matplotlib.
- **User Interfaces**: The emotion classifier offers a web interface, and the voice-based analyzer has a GUI.
- **Real-time Updates**: The real-time face emotion detection application offers real-time feedback based on facial expressions.

The **MoodTracker** project is designed to help users gain insights into their emotional well-being and better understand their daily mood patterns.

---
