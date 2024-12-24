# Fake News Detector

This project is a full-stack application that detects whether a news article is fake or factual. It includes a backend built with Flask and a frontend developed using React/Next.js.

---

## Features

- **Fake News Detection**:
  - Predicts whether a given news article is fake or factual using machine learning.
- **Endpoints**:
  - `/`: Health check endpoint.
  - `/api/predict`: Accepts a JSON payload with the text of the news article and returns the prediction.
  - `/api/data`: Provides sample data for testing.
- **Frontend Integration**:
  - A user-friendly interface to input news text and view predictions.

---

## Tech Stack

### Backend
- Flask
- scikit-learn
- joblib
- pandas

### Frontend
- React/Next.js
- Axios

---

### Clone the Repository
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
