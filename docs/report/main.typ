#import "template.typ": article

#show: article.with(
    title: "Forecasting Student Success through Artificial Intelligence: A Case Study Using Neural Networks",
    authors: (
      "Author One": (
        "affiliation": "CC",
        "name": "Dongli Liu"
      ),
      "Author Two": (
        "affiliation": "CC",
        "name": "Wendy Paraizo"
      ),
      "Author Three": (
        "affiliation": "CC",
        "name": "Daniel Ifejika"
      )
    ),
    affiliations: (
      "CC": "Engineering School of Centennial College. 941 Progress Ave, Scarborough, ON M1G 3T8 CA"
    ),
    abstract: [Accurately forecasting student success is crucial for improving retention and outcomes. This study predicts first-year persistence using Artificial Intelligence (AI) and Neural Networks (NN). A real-world dataset is processed to address challenges like missing values and class imbalance. The NN model, optimized with advanced techniques, outperforms traditional methods in predicting persistence. While the results show promise, the study also highlights challenges in model interpretability and generalization, suggesting areas for future exploration in AI-driven educational interventions.],
    keywords: ("Artificial Intelligence", "Neural Network", "Student Success", "Predictive Analytics"),
    bib: "main.bib",
    appendix: include "appendix.typ"
)

= Introduction

Student success is a critical concern for educational institutions, influencing graduation rates, career outcomes, and societal contributions. However, predicting student success is complex, as factors such as socio-economic background, prior academic performance, and personal circumstances can significantly affect outcomes. Institutions face challenges in identifying students at risk of underperforming or dropping out, making it essential to develop methods for early intervention.

Programs like the Helping Youth Pursue Education (HYPE)#footnote([The HYPE program at Centennial College has given many young people new confidence to return to school by helping to overcome the economic and social barriers that may have interfered with school attendance in the past and by providing a nurturing, inclusive environment for youth aged 17-29, primarily living in the underserved neighbourhoods of Toronto.])#cite(<armstrong2017>)#cite(<hypecentennial>) initiative at Centennial College have demonstrated the importance of targeted support for students from underserved communities#cite(<maher2013>). HYPE’s focus on addressing barriers to post-secondary education for youth from disadvantaged backgrounds highlights the need for data-driven strategies to improve retention and success rates in higher education@armstrong2017.

Artificial Intelligence (AI), particularly Neural Networks (NN), offers promising solutions in education. Neural networks can analyze large datasets, uncovering complex patterns that traditional methods may miss. This study uses AI to predict first-year persistence, a key indicator of student success, using a real-world dataset. The goal is to demonstrate how neural networks can improve prediction accuracy and provide insights for better supporting students, while addressing challenges such as missing data, class imbalance, and insufficient amount of data.

= Methodology

== Dataset Preprocessing

The dataset used in this study originates from Centennial College, specifically from engineering school, and contains data on first-year students, including socio-economic status, high school GPA, attendance records, participation in extracurricular activities, and other features hypothesized to influence student success, as shown in @table-data-info.

#figure(
  table(
    columns: (1fr, auto,auto),
    stroke: none,
    align: left,
    [*Column*], [*Non Null*], [*Dtype*],
    [First Term Gpa], [1420], [float64],
    [Second Term Gpa], [1277], [float64],
    [First Language], [1326], [float64],
    [Funding], [1437], [int64],
    [School], [1437], [int64],
    [Fast Track], [1437], [int64],
    [Coop], [1437], [int64],
    [Residency], [1437], [int64],
    [Gender], [1437], [int64],
    [Previous Education], [1433], [float64],
    [Age Group], [1433], [float64],
    [High School Avg Mark], [694], [float64],
    [Math Score], [975], [float64],
    [English Grade], [1392], [float64],
    [FirstYearPersistence], [1437], [int64],
  ),
  caption: [Information of the Raw Data]
)<table-data-info>

*Missing values* were handled by applying `dropna` columns with a small portion of null values. For the remaining missing data, `IterativeImputer` with `RandomForestRegressor()` was used to infer values, preserving feature relationships. This method was particularly important for columns like _High School Average Marks_, which had over 50% missing values, as using mean imputation could have introduced significant bias #cite(<sklearn>). 

#figure(
  {
    ```python
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(),
        max_iter=10,
        n_nearest_features=6,
        imputation_order="ascending", 
        tol=1)
    imputed = imputer.fit_transform(dropped)
    ```
  },
  caption: [`sklearn.impute.IterateImputer` is a experimental class that infers values by iteratively modelling a function with other features. It is very useful for _multivariate_ problems. @sklearn]
)<code-impute>

The data is significantly imbalanced, with 1138 positive while only 299 negative instances.To address *class imbalance*, _upsampling_ was used instead of _downsampling_ to retain valuable information @sklearn. This approach was also chosen because the dataset is relatively small, and maintaining as much data as possible is crucial for training an effective model.

== Input Pipeline

The `tf.data.Dataset` API was utilized to create a descriptive and efficient input pipeline #cite(<tensorflow>). While the dataset used in this project is not large, the choice to use this API aligns with practical considerations, such as the potential deployment of the model for both _prediction_ and _online machine training_#cite(<wikionline>)#footnote([online machine training: a method of machine learning in which data becomes available in a sequential order and is used to update the best predictor for future data at each step.]). By leveraging the capabilities of the API `tf.data.Dataset`, an efficient data processing chain was constructed following the approach detailed by #cite(<geron2022>, form: "author"). 

The dataset was split into three subsets: `train_set`, `val_set`, `test_set` using the `take()` and `skip()` functions provided by the API. The `train_set` comprises 70% of the data, while the `val_set` and `test_set` each consist of 15%. 

== Neural Network Architecture

The neural network model used in this study is designed to predict _first-year persistence_. The architecture consists of an input layer, two hidden layers, and an output layer, with each hidden layer followed by a `BatchNormalization` layer and a `Dropout` layer for _normalization_ and _regularization_. As shown in @fig-model-arch. 

The input layer is simply using a `Flatten()` to accept dataset. The hidden layers consist of 256 neurons in the first layer and 128 neurons in the second, chosen based on the experiments in a `for` loop. Each layer uses the _Rectified Linear Unit_ (`relu`) activation function, which is known to handle the _vanishing gradient_ problem better than traditional `sigmoid` functions. The output layer consists of 2 neuron, using the `softmax` activation function to adapt the `one-hot` format of dataset from the input pipeline, predicting the probability of student persistence. `Dropout` _regularization_ techniques were applied in the hidden layers to prevent _overfitting_, with a dropout rate of 0.2. The model is trained using the _adam optimizer_, which combines the advantages of both _AdaGrad_ and _RMSProp_ for faster convergence and improved performance in _sparse gradients_.

#figure(
  image("../images/model_summary.png"),
  caption: [Architecture of the Selected Model]
)<fig-model-arch>

== Model Training

The model was designed to train for 100 epochs, with `EarlyStopping` implemented to prevent _overfitting_ if the validation loss does not improve after 10 epochs, and stopped the training at Epoch 45. A `ModelCheckpoint` was set to monitor the `val_accuracy` to capture the best weights.

#figure(
  {
```python
history = model.fit(
  train_set.repeat(),
  epochs=100,
  validation_data=val_set.repeat(),
  steps_per_epoch=1000,
  validation_steps=100,
  callbacks=[mc, es])
```
  },
  caption: [Train model using infinite-length dataset with finite steps.]
)

During training, the _learning curve_#footnote([learning curve: refers to a graphical representation of the model's performance during training, typically showing metrics like accuracy and loss over training epochs]) was unstable due to the small dataset and the relatively complex model. To address this issue, the `repeat()` function was used to create an *infinite-length dataset*, allowing for extended training. The parameters `steps_per_epoch` and `validation_steps` were set accordingly to ensure the model received sufficient training despite the dataset's limited size. According to the `accuracy` and `loss` recorded in the training history, this approach significantly smoothed the _learning curve_. The model achieved high accuracy as well as better generalization capability compared to the regular training approach. 

#figure(
  kind: table,
  table(
    columns: 2,
    stroke: none,
    align: left,
    [`accuracy`],[0.9877],
    [`val_accuracy`], [1.0000],
    [`loss`], [0.0354],
    [`val_loss`], [0.0033]
  ),
  caption: [Best accuracy and loss in training]
)<table-train-metrics>

From @table-train-metrics, When the train was early stopped at Epoch 45, the captured model reaches a 100% validation accuracy. 

#figure(
  caption: [The learning curves show that using an infinite-length dataset (bottom figures) results in higher accuracy, lower loss, and greater stability, with a consistent gap between training and validation curves, indicating better generalization and reduced overfitting.],
  grid(
    columns: 2,
    rows: 2,  
    [#image("../images/256-128-0.2-norepeat_acc.png")],
    [#image("../images/256-128-0.2-norepeat_loss.png")],
    [#image("../images/256-128-0.2-repeat_acc.png")],
    [#image("../images/256-128-0.2-repeat_loss.png")],
  )
)

== Evaluation Metrics

The performance of the neural network model was first evaluated by a straightforward `model.evaluate()` function to the unseen `test_set` and reached a astonishing accuracy.

#figure(
  kind: table,
  table(
    columns: 2,
    stroke: none,
    align: left,
    [dataset],[1000 steps infinite test_set],
    [`loss`], [0.003268422558903694],
    [`accuracy`], [0.9997187256813049],
  ),
  caption: [Evaluation with test_set. The repeat() was used again because the single `test_set` consistently produced 100% accuracy.]
)<table-test-evaluate>

This outcome reflects the model's excellent performance on unseen test data. In this case, the _confusion matrix_(@fig-confusion-matrix) and _ROC AUC_ seems unnecessary because they provide little additional insight when the model consistently predicts the correct class for all instances, resulting in perfect metrics.

#figure(
  caption: [A perfect confusion matrix implies that the model correctly classified all instances, with no false positives or false negatives.],
  image("../images/confusion_matrix.png")
)<fig-confusion-matrix>

#figure(
  caption: [An AUC of 1.00 means that the model has perfect discrimination ability, correctly distinguishing between all positive and negative instances. It indicates that the model's predictions are flawless, with no false positives or false negatives, leading to a perfect classification performance across all thresholds.],
  image("../images/roc_auc.png")
)<fig-roc-auc>

== Predictive Website

A full-stack website was built to utilize the model, featuring a Node.js frontend and Flask backend. The backend includes a simple `.csv` file as a database layer, used to append data during _online training_. While the website currently only supports prediction, the backend also developed to support two additional functionalities including train_one and train_batch. More information can be found in Appendix.

#figure(
  grid(
    rows: 3,
    [#image("../images/predict.png")],
    [#image("../images/train_one.png")],
    [#image("../images/train_batch.png")],
  ),
  caption: [Diagrams showing the process of handling the 3 functionalities in backend. Multiple `@tf.function` was used to effectively handle the flow, matching the need of quick response. If the project have chance to be continued, _Tensorflow Serving_ can be used to manage the model versions.]
)

= Discussion

== Effectiveness in Prediction

The neural network model demonstrated impressive performance in predicting first-year student persistence, with an accuracy of 100% on the validation set and nearly perfect results on the test set. This aligns with the growing body of research suggesting that AI, particularly neural networks, can provide valuable insights into student success predictions, outperforming traditional statistical methods. 

== Model Interpretability

One of the key challenges highlighted by this study is the interpretability of neural network models. While the model provides high accuracy, understanding the specific factors driving its predictions remains complex. In educational settings, stakeholders such as instructors, counselors, and administrators may require more transparency in decision-making models. 

== Class Imbalance

Addressing class imbalance through upsampling proved to be effective in maintaining valuable data points, but it also raised questions about potential biases introduced by this technique. While upsampling prevented the loss of information, it might have artificially influenced the distribution of the data, potentially skewing the model's predictions.

== Data Quality

Although the missing values were handled gracefully with `IterativeImputer` to preserve relationships between features, the importance of completeness of data is still worth to be emphasized. Missing data is often a recurring issue in real-world datasets, and the choice of imputation technique significantly impacts model accuracy. To handle missing values, other imputation methods such as _k-nearest neighbors_ may have similar or better performance. The variance of the data source may also lead to a biased model, for example, all the instances of this dataset are from Engineering school. If the data is more variable, the model will have chance to learn better generalization skill.

== Regularization

The model's _early stopping_ and _dropout_ regularization techniques contributed to minimizing overfitting, which is a common challenge when working with small datasets. Despite the model's high accuracy on unseen test_set, there may still be concerns regarding its generalization to other datasets with different characteristics. Further research is needed to test the model’s robustness across a broader set of educational institutions and student demographics.

== Model Complexity
Since we did not penalize the model's complexity during the design phase, the neural network evolved into a relatively complex structure. Although it is not overfitting, such complexity may increase computational costs and reduce interpretability. Future work could involve applying regularization, simplifying the architecture, or exploring model pruning to balance performance, efficiency, and scalability.

== Model Tuning

Since the model achieved perfect performance without signs of _overfitting_ after optimizing the network architecture, further tuning techniques were not explored in this project. However, future studies could consider strategies such as hyperparameter optimization methods like _Grid Search_ and _Random Search_, applying regularization techniques like L1 and L2, and experimenting with diverse _units_, _activation_ functions, and _loss_ functions to enhance the model’s robustness and generalizability.

== Broader Generalization

Although the model shows excellent performance in this dataset, the effectiveness of applying the model to the broader use case, for example, other institutions or schools are uncertain. In that case, more evaluation should be performed, and more diverse data are probably required to further train the model. 

= Conclusion

The study reveals neural networks' powerful capability to predict first-year student persistence with near-perfect accuracy. Key findings include the model's strong predictive performance balanced against significant challenges such as interpretability limitations. The research addressed data preprocessing through advanced techniques like upsampling and iterative imputation, while implementing regularization methods such as dropout and early stopping to mitigate overfitting risks. While the model achieved perfect performance on this dataset, and should be helpful to the HYPE program@armstrong2017, its effectiveness in broader scenarios remains unproven due to potential variations in student populations, institutional contexts, and data quality. 

Critical insights emerge from the analysis: while AI can effectively forecast student success, educational stakeholders may require transparent models that enable clear, actionable decision-making. Future work should focus on expanding dataset diversity and advancing explainable AI techniques to bridge the gap between sophisticated predictive capabilities and meaningful educational interventions.