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
      "CC": "Progress Campus of Centennial College. 941 Progress Ave, Scarborough, ON M1G 3T8 CA"
    ),
    abstract: [Accurately forecasting student success is crucial for improving retention and outcomes. This study predicts first-year persistence using Artificial Intelligence (AI) and Neural Networks (NN). A real-world dataset is processed to address challenges like missing values and insufficient amount of data. The NN model, optimized with advanced techniques, outperforms traditional methods in predicting persistence. While the results show promise, the study also highlights challenges in model interpretability and generalization, suggesting areas for future exploration in AI-driven educational interventions.],
    keywords: ("Artificial Intelligence", "Neural Network", "Student Success", "Predictive Analytics"),
    bib: "main.bib",
)

= Introduction

Student success is a critical concern for educational institutions, influencing graduation rates, career outcomes, and societal contributions. However, predicting student success is complex, as factors such as socio-economic background, prior academic performance, and personal circumstances can significantly affect outcomes. Institutions face challenges in identifying students at risk of underperforming or dropping out, making it essential to develop methods for early intervention.

Programs like the Helping Youth Pursue Education (HYPE)#cite(<armstrong2017>) initiative at Centennial College have demonstrated the importance of targeted support for students from underserved communities #cite(<maher2013>). HYPE’s focus on addressing barriers to post-secondary education for youth from disadvantaged backgrounds highlights the need for data-driven strategies to improve retention and success rates in higher education @armstrong2017.

Artificial Intelligence (AI), particularly Neural Networks (NN), offers promising solutions in education. Neural networks can analyze large datasets, uncovering complex patterns that traditional methods may miss. This study uses AI to predict first-year persistence, a key indicator of student success, using a real-world dataset. The goal is to demonstrate how neural networks can improve prediction accuracy and provide insights for better supporting students, while addressing challenges such as missing values and insufficient amount of data.


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

=== Challenge of and solution of the dataset

- Missing Values.

Missing values were handled through imputation techniques, with mean imputation used for numerical features and the mode for categorical features. Additionally, the dataset was balanced using under-sampling techniques to ensure that the model was not biased toward the majority class. Feature engineering was applied to create new features that might improve the model’s predictive performance, such as interaction terms between attendance and GPA, which are hypothesized to have an additive effect on persistence.
