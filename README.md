# Student Persistence Full-Stack Intelligent App

## How to interact with `main` branch

> to maintain a clean repository, direct `push` to `main` thread is no longer supported. Instead, simply create a [pull request](https://github.com/Dongli99/ml-nn-studentpersistence/pulls).Team members will review, comment, and merge it to main. check [workflow.md](./docs/workflow.md) for more details.

## Run Website

```bash
python web/back/app.py
cd web/front
npm start
```

## Functionality & Endpoints

| Endpoint      | Method | Description                                           |
|---------------|--------|-------------------------------------------------------|
| `/predict`    | POST   | Accepts a single data point and returns the prediction. |
| `/train-one`  | POST   | Accepts a single labeled data point and trains the model. |
| `/train-batch`| POST   | Accepts a CSV file containing labeled data and trains the model. |

## Sequence Diagram

![alt Sequence Diagram](/docs/images/sequence.png)
