from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS, cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
@cross_origin()
def home_page():
    # Redirect to /predict when accessing the home page
    return redirect(url_for('predict_datapoint'))

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get data from form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )

            # Convert form data to DataFrame
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            # Make prediction
            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")

            # Render the results
            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template('home.html', error="An error occurred during prediction. Please check your inputs.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
