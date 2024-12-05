import React from 'react';

const PredictionResults = ({ result }) => {
  // Check if the result is null or undefined
  if (!result) {
    return <p>No prediction result to display</p>;
  }

  // Assuming result is an object with prediction data
  return (
    <div>
      <h3>Prediction Result:</h3>
      <p>{JSON.stringify(result)}</p>
      {/* You can display individual fields of the result if it's an object */}
      {/* For example: */}
      {/* <p>Prediction: {result.prediction}</p> */}
    </div>
  );
};

export default PredictionResults;
