# Cricket Score Prediction using XGBoost Regression

This project involves a regression-based approach to predict the batting score of a team in One Day Internationals (ODIs). The model is trained on ODI cricket data and provides predictions based on the given dataset.

## Features

- **Regression Model**: Predicts the batting score of a team.
- **Trained on ODI Data**: The dataset used is specific to ODIs.
- **Customizable**: To use this model for other formats or datasets, simply update the dataset and file path.

## How to Use

1. **Clone the Repository**:
   
   git clone https://github.com/AMevans12/Machine-Learning.git
   cd Machine-Learning/Cric-Sheet
   

2. **Prepare the Environment**:
   Install the required Python libraries:
   
   pip install -r requirements.txt
   

3. **Run the Code**:
   Execute the notebook or script:
   
   jupyter notebook CEP.ipynb
   

4. **Update Dataset**:
   If you want to train the model on a different dataset, replace the dataset file in the `file_path` variable within the notebook.

## Files

- `CEP.ipynb`: Main notebook for score prediction.
- `requirements.txt`: List of dependencies.

## Notes

- The model is specifically trained on ODI data. For other formats or datasets, update the `file_path` variable in the notebook to point to the new dataset.

## Example

Below is an example of how the model predicts a score:


# Load the dataset
file_path = "path_to_your_dataset"

# Run the prediction
predicted_score = model.predict(input_features)
print(f"Predicted Score: {predicted_score}")


## Contribution
Feel free to open issues or submit pull requests to improve the project.

## License
This project is open-source and available under the MIT License.
