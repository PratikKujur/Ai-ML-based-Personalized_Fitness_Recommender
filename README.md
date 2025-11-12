# 💪 AI-ML Based Personalized Fitness Recommender

## About the Project

The **AI-ML Based Personalized Fitness Recommender** is an intelligent system that provides personalized exercise recommendations based on individual fitness profiles and habits. This project leverages machine learning algorithms (specifically XGBoost) to analyze user characteristics and recommend the most suitable exercises from a curated set of 50+ exercises.

### Key Features
- **Personalized Recommendations**: Get top 10 exercise recommendations tailored to your profile
- **Comprehensive User Analysis**: Considers demographic data, fitness level, dietary preferences, and equipment availability
- **XGBoost Model**: Uses gradient boosting for accurate predictions
- **Streamlit Interface**: User-friendly web interface for easy interaction
- **AI-Powered Insights**: Integrates with Google Gemini API for intelligent explanations of recommendations

---

## Project Structure

```
Ai-ML-based-Personalized_Fitness_Recommender/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
├── LICENSE                         # License file
│
├── dataset/
│   ├── Final_data.csv             # Main training dataset
│   └── meal_metadata.csv          # Meal-related metadata
│
├── models/
│   └── exercise_recommender.pkl   # Trained XGBoost model
│
├── src/
│   ├── __init__.py                # Package initializer
│   ├── constants.py               # Configuration constants & exercise mappings
│   ├── model_pipeline.py          # ML pipeline for model training
│   ├── utils.py                   # Utility functions for preprocessing
│   └── __pycache__/               # Compiled Python files
│
├── experiment/
│   └── exp.ipynb                  # Jupyter notebook for experimentation
│
└── __pycache__/                   # Compiled Python cache

```

### Key Directories

- **`src/`**: Contains core project modules
  - `constants.py`: Exercise mappings and model parameters
  - `model_pipeline.py`: Machine learning pipeline for training and data transformation
  - `utils.py`: Preprocessing and utility functions
  
- **`dataset/`**: Contains training and metadata files
  - `Final_data.csv`: Primary dataset with user fitness data and exercise labels
  - `meal_metadata.csv`: Nutritional metadata for meal recommendations
  
- **`models/`**: Stores the trained XGBoost model
  
- **`experiment/`**: Jupyter notebooks for model exploration and experimentation

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/PratikKujur/Ai-ML-based-Personalized_Fitness_Recommender.git
cd Ai-ML-based-Personalized_Fitness_Recommender
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root directory and add your Google Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

Or configure it via Streamlit secrets (`.streamlit/secrets.toml`):

```toml
GEMINI_API_KEY = "your_api_key_here"
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`.

---

## How to Use

1. **Fill in Your Profile**: Enter your personal information including:
   - Age, Height, Weight
   - Gender and Experience Level
   - Dietary preferences
   - Available equipment
   - Workout frequency and session duration

2. **Get Recommendations**: Click the "Get Recommendations" button

3. **View Results**: The system will display:
   - Top 10 personalized exercise recommendations
   - Difficulty levels for each exercise
   - AI-powered explanations from Google Gemini

---

## Technical Details

### Model Architecture
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Features**: 
  - Categorical: Gender, Diet Type, Cooking Method, Equipment Needed, Difficulty Level, Body Part
  - Numerical: Age, Water Intake, Height, Weight, Session Duration, Workout Frequency, Experience Level, Daily Meal Frequency
- **Target**: Exercise type (50+ exercises)

### Data Processing Pipeline
1. **Data Loading**: Reads from `Final_data.csv` using Kaggle Hub API
2. **Feature Engineering**: Categorical encoding and numerical scaling
3. **Preprocessing**: Handles missing values and feature normalization
4. **Model Training**: Trains XGBoost classifier on processed data
5. **Model Serialization**: Saves trained model as pickle file

### Dependencies
- `streamlit`: Web interface framework
- `scikit-learn`: Machine learning utilities
- `xgboost`: Gradient boosting model
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `google-genai`: Google Gemini API integration
- `joblib`: Model serialization

---

## Dataset

The project uses a lifestyle dataset containing:
- **User Profiles**: Demographics, fitness metrics, and preferences
- **Exercise Data**: 50+ different exercises with metadata
- **Target Variable**: Recommended exercises for each user profile

All data is sourced from Kaggle through the `kagglehub` library.

---

## Model Training

To retrain the model:

```python
from src.model_pipeline import pipline

pipeline = pipline()
pipeline.get_data()           # Download dataset
pipeline.data_transforma()    # Transform and preprocess data
pipeline.model_train()        # Train XGBoost model
```

---

## License

This project is licensed under the terms specified in the `LICENSE` file.

---

## Author

**Pratik Kujur**

For questions or contributions, please open an issue or pull request on the GitHub repository.

---

## Future Enhancements

- [ ] Add meal recommendations based on dietary preferences
- [ ] Implement user feedback loop for model improvement
- [ ] Add progress tracking and workout history
- [ ] Integrate with fitness tracking devices
- [ ] Deploy on cloud platforms (AWS, GCP, Azure)
- [ ] Add multi-language support

---

## Troubleshooting

### Issue: Model file not found
**Solution**: Ensure the `models/exercise_recommender.pkl` file exists. If not, retrain the model using the pipeline.

### Issue: API key errors
**Solution**: Verify your `GEMINI_API_KEY` is correctly set in the `.env` file or Streamlit secrets.

### Issue: Dataset download fails
**Solution**: Check your internet connection and ensure you have `kagglehub` properly configured.

---

## Support

For support, please open an issue on the GitHub repository or contact the project maintainer.
