# Sentiment Analysis API

This API provides sentiment analysis for English text using a machine learning model. It classifies text into sentiment categories based on the trained model.

## Features

- Text sentiment prediction via web interface or API calls
- Text preprocessing
- Simple health check endpoint
- Web interface with HTML forms
- REST API endpoints

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface form |
| `/predict` | POST | Process sentiment analysis form |
| `/health` | GET | Health check |

## Installation

### Prerequisites

- docker

### Steps

1. Clone the repository:
```bash
git clone https://github.com/your-repo/sentiment-analysis-api.git
cd sentiment-analysis-api
```
2. Build and up containers:
```bash
docker-compose build
docker-compose up
```
Ready to start!
