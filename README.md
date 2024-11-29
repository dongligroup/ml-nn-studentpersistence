# Student Persistence Full-Stack Intelligent App

## Functionality & Endpoints

| Endpoint      | Method | Description                                           |
|---------------|--------|-------------------------------------------------------|
| `/predict`    | POST   | Accepts a single data point and returns the prediction. |
| `/train-one`  | POST   | Accepts a single labeled data point and trains the model. |
| `/train-batch`| POST   | Accepts a CSV file containing labeled data and trains the model. |

## Sequence Diagram

![alt Sequence Diagram](/images/sequence.png)
