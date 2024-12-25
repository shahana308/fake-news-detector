# Fake News Detector

## Overview
The Fake News Detector is an AI-powered web application designed to classify news articles as factual or fake. It leverages advanced Natural Language Processing (NLP) and machine learning techniques to provide reliable predictions.

## Features
- Input news headlines or articles to predict their authenticity.
- Uses Natural Language Processing (NLP) and machine learning for classification.
- Displays results in a user-friendly interface.

## Setup

### Backend
1. Clone the repository.
2. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
3. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the backend server:
   ```bash
   gunicorn app:app --bind 0.0.0.0:5000
   ```

### Frontend
1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## Deployment

### Backend
- Hosted on [Render](https://render.com).
- Ensure the start command is set to:
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT
  ```

### Frontend
- Deployed on [Vercel](https://vercel.com).
- Deploy the `frontend` folder.

## Usage
Visit the deployed application: [Fake News Detector](https://fake-news-detector-ochre.vercel.app/)

1. Enter a news headline or article in the text box.
2. Click "Check News."
3. View the prediction (Factual News or Fake News).

## Future Enhancements
- Regular updates to the dataset with the latest news.
- Integration with real-time news APIs.
- Improved model accuracy with more training data.

---

