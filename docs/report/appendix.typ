= Appendix

== Backend Guidelines

*1. API Layer (`app.py`)*

*Endpoints*:
  - `/predict (POST)`: Accepts a `JSON` payload containing student features, converts the data into a Student object, validates it, and generates predictions, returning rounded predictions as a `JSON` response.
  - `/train-one (POST)`: Accepts a single labeled data point for incremental training, enabling real-time model adaptability without requiring batch training.
  - `/train-batch (POST)`: Accepts `CSV` files for bulk training, designed for periodic model updates using larger datasets.

*2. Core Logic (`core.py`)*

*Model Management*: Dynamically loads models based on the environment (development or production), ensures seamless switching between testing and deployment, and uses TensorFlow's load_model for robust model handling.

*Data Handling*: Supports numerical and categorical data processing, where numerical features are normalized using predefined means and standard deviations, and categorical features are encoded using one-hot encoding for model compatibility. Incorporates a CSV-based training pipeline with optimized TensorFlow handling, using the tf.data API for efficient batch processing and multithreading, skipping headers and shuffling data to improve model generalization. Also validates data to ensure consistency and prevent model errors.

*Prediction*: Processes student features into a compatible input format, generates predictions with TensorFlow, and isolates probabilities for interpretability.

*Training*: Supports both incremental (train_one) and batch (train_batch) training, encapsulates preprocessing, data augmentation, and error handling during model fitting, and periodically saves the model after successful training.

*Validation*: Validates input data against metadata (e.g., ranges for numerical fields, allowed values for categorical fields) and prevents inconsistent or invalid inputs from corrupting the model.


*3. Student Object*

*Structure*: Encapsulates all student attributes, including academic and demographic features, and converts raw data into normalized, validated inputs for the model.

*Validation*: Checks numerical ranges and categorical mappings using metadata, ensuring the integrity of predictions and training by rejecting invalid data.

*Conversion*: Converts JSON payloads into Student objects for easy handling, and generates CSV strings for integration with TensorFlow pipelines.

== Frontend Guidelines
*1. `PredictionForm.js`*

Renders a form to collect student data and submit predictions.Uses `React`’s `useState` hook for state management to track form inputs and results, implements client-side validation to ensure required fields are filled and numeric inputs are valid, shows a loading spinner and dynamic messages for predictions, integrates with the backend via `axios`, and is customizable with reusable components.

*2. `PredictionResults.js`*

Displays prediction results. Renders results only when available, and supports future expansions for more metrics.

*3. `PredictionForm.css`*

Styles the `PredictionForm.js` component. Leverages modern design trends with gradients, rounded borders, and responsive layout, highlights errors with distinct colors and styles, and provides hover effects on buttons for interactivity.