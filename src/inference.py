import joblib
from data_preprocessing import preprocess_text, vectorize_text

# Load the SVM model to reduce inference latency
svm_model = joblib.load('svm_model.joblib')

def predict(input_text, model):
    """Perform preprocessing and then a prediction using the model."""
    preprocessed_text = preprocess_text(input_text)
    vectorized_text = vectorize_text(preprocessed_text)
    
    return svm_model.predict(vectorized_text)[0]

def handler(event, context):
    """Handle the Lambda invocation."""
    # Extract the input text from the event
    input_text = event['text']
    if input_text is None:
        # Return an error message
        return {
            'statusCode': 400,
            'body': 'No text was provided.'
        }
    
    try:
        # Perform inference
        prediction = predict(input_text, svm_model)
        # Return the prediction
        return {
            'statusCode': 200,
            'body': prediction
        }
    except:
        # Return an error message
        return {
            'statusCode': 500,
            'body': 'An error occurred during inference.'
        }