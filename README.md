# Overview
I am a data scientist and software engineer specializing in extracting actionable intelligence from data. With a strong academic background in mathematics and data science, I am transitioning into a data science career, and I am excited to join a mission-driven company.

I am an expert in modern data science practices, modern AI practices, agentic workflows, and machine learning infrastructure and implementation. I am passionate about coding with an emphasis on craftsmanship, clean code, code readability, and documentation.

I am well versed in the Python data science ecosystem and data science infrastructure and design. I am adept at data curation, data visualization, and business intelligence. My experience is broad, with knowledge of LLMs, computer vision, natural language processing, data wrangling, and a wide variety of data modeling techniques.

I showcase my work publicly on Github. I enjoy tackling challenging problems that force me to expand my repertoire and learn new technologies.

# Projects
## Agentic and Neural Network Projects

### ModelBot Data Science Tool

Repository: [ModelBot](https://github.com/JoshuaGottlieb/TinyTroupeSimulation/tree/main/src/deliverable-03)
<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>Developed an Agentic AI chat program that guides a user with no data science expertise in performing regression and classification on a CSV data set of their choosing.</li>
    <li>The script creates a chat between the user and an LLM, and upon the invocation of certain key phrases, begins the process of loading and preprocessing the data (cleaning values, changing data types, validating and replacing missing values), creating a model or set of models (including automatic cross-validation and hyperparameter tuning), and generating a PDF business report geared towards explaining the model performance to a non-technical user.</li>
    <li>The ModelBot program is designed to be user-friendly, allowing for a guided experience and automatically calls appropriate functions recursively to ensure no steps are executed out of order. The program is designed with an API endpoint mockup using FastAPI to run a local server, in order to mimic real-world business practices of offloading processing to a server and keeping the chat contained locally. Using this API endpoint, data is serialized and deserialized to allow passage of model and data artifacts using HTTP protocols.</li>
    <li>While the current version of ModelBot is limited to performing regression (Linear Regression, Polynomial Ridge Regression, and ElasticNet Regression) and classification (Logistic Regression, Decision Trees, Random Forest) tasks, the architecture is designed such that additional functions can be added as API endpoints by including the new function metadata in JSON form without modifying the main logic structures governing function execution and chat capabilities. The repository contains a detailed system design document covering the ability of ModelBot to utilize a directed function dependency graph encoded as JSON metadata in order to perform recursive function execution, allowing the program to perform the task specified by the user without the user requiring any underlying knowledge of function dependencies or requirements.</li>
  </ul>    
</details>

### Facial Recognition with Deepface

Repository: [Facial Recognition with Deepface](https://github.com/JoshuaGottlieb/Facial-Recognition-with-Deepface)
<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>Used Facebook's DeepFace neural net to create a facial recognition application.</li>
    <li>Utilized the Labeled Faces in the Wild dataset to demonstrate the ability of the DeepFace neural net.</li>
    <li>Identified cases that could cause DeepFace to fail and steps to improve the effectiveness of a pre-trained facial recognition model through careful preprocessing and selection of input images.</li>
    <li>Developed image preprocessing code to automatically align, extract and resize the image to contain only the facial area and eliminate the background using the dlib, RetinaFace, and MediaPipe libraries in conjunction with OpenCV.</li>
    <li>Due to the vast amount of training data needed to train the data from scratch, this project used the pretrained weights developed by Swarup Ghosh and emulated his preprocessing steps. Ghosh trained his neural network based on the DeepFace neural net described in the original paper published by Facebook.</li>
    <li>Evaluated the effectiveness of the pre-trained neural net on the Labeled Faces in the Wild (LFW) dataset at different thresholds which each maximized different classification metrics (Balanced Accuracy, Precision, Recall, F1-Score, Markedness, and Mathew's Correlation Coefficient). Under the threshold with the highest F1-Score, the DeepFace neural net was able to attain a precision of 60.5% at the cost of only being able to attain a recall of 25.8% on the LFW dataset.</li>
    <li>Identified problems using unstructured data which caused false positives and false negatives, such as uncontrolled lighting, poor image alignment and cropping, and the lack of frontalization of faces (when the subject is not looking at the camera). Created a small personal dataset with more controlled conditions and illustrated great leaps in performance by the same pre-trained classifier. On a personal dataset, the threshold with the highest F1-Score produced a precision of 55.8% and recall of 65%, a noticeable improvement upon the unconstrained LFW dataset, showcasing that careful control of input images greatly changes the effectiveness of a facial recognition neural network.</li>
    <li>Identified potential improvements to the preprocessing pipeline used by Ghosh to enhance the effectiveness of the DeepFace neural net without changing the underlying neural net architecture.</li>
  </ul>    
</details>

### Sketch to Image Generation Using Conditional GAN

Repository: [Rest of the Owl](https://github.com/JoshuaGottlieb/Rest-of-the-Owl)
<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>A Tensorflow neural network project to generate highly detailed sketches from minimal sketches of owls.</li>
    <li>Used the Python Imaging Library (PIL) and OpenCV libraries to manipulate and preprocess images.</li>
    <li>Applied the Extended Difference of Gaussians algorithm to generate sketches from ground-truth images. This method consists of using a combination of strong and weak Gaussian blurred versions of the image to extract edges from images. Differencing the Gaussian blurred versions and applying thresholding resulted in images with high edge definition and lower overall detail.</li>
    <li>Some images were white sketches on black backgrounds instead of the usual black sketches on white backgrounds. Images were automatically normalized to be black images on white backgrounds by designating the outer 20% of the image as "border" and calculating how much of this border was black - images with borders above a certain threshold were likely white images on black backgrounds and were inverted.</li>
    <li>Employed Tensorflow's Functional Network API to create a conditional generative adversarial network (cGAN), implementing a U-net architecture for the generator and a patchGAN architecture for the discriminator.</li>
    <li>Wrote custom training and fit loops for the neural network using Tensorflow 2.x architecture, translating from Tensorflow 1.x architecture. Trained the model in Google Colab using 1,905 training samples.</li>
    <li>The generator and discriminator were evaluated separately using their appropriate loss functions, and the overall strength of the model was tested using the common metric called the Frechet Inception Distance (FID). For generative algorithms, it is difficult to capture performance solely through a metric, and so the images were inspected manually. In most cases, the model was able to closely recreate the original images from the minimal sketches, even when using images of other subjects such as cats or humans.</li>
  </ul>    
</details>

## Data Wrangling and Database Projects

### NY Crime Data SQL

Repository: [NY Crime Data SQL](https://github.com/JoshuaGottlieb/NY-Crime-Data-SQL)

<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>A relational database project utilizing NYC crime data from NYC OpenData.</li>
    <li>Data was cleaned using Pandas and partitioned into multiple SQL tables using the SQLite package.</li>
    <li>Executed queries to perform analysis of the crime data, including extracting the top 5 crime types, ratio of complaints to arrests, per capita crime statistics, and crime severity statistics by borough, as well as victim versus suspect/perpetrator crime demographics, and the performance of police precincts by year.</li>
  </ul>
</details>

### AirBnB Graph Database with Neo4j

Repository: [AirBnB-Neo4j](https://github.com/JoshuaGottlieb/AirBnB-Neo4j)

<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>A graph database project utilizing AirBnB listing data to populate nodes and relations using Neo4j through the neo4j GraphDatabase drive API.</li>
    <li>Wrote queries using the Cypher graph query language to illustrate the potential of the database for recommendation systems, customer profiling, and marketing analysis.</li>
  </ul>
</details>

## Classic EDA and Modeling Machine Learning Projects

### Music Genre Predictions by Song Lyrics Using Word2Vec and Topic Modeling
Repository: [NLP-Genre-Classification](https://github.com/JoshuaGottlieb/NLP-Genre-Classification)

<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>Classification project using NLP Bag-of-Words representation and Word2Vec to predict song genre using lyrics.</li>
    <li>Obtained song metadata from musiXmatch's API: https://developer.musixmatch.com.</li>
    <li>Scraped over 100,000 songs from songlyrics.com using BeautifulSoup, obtaining over 40,000 usable records. Used REGEX to clean the data.</li>
    <li>Performed topic modeling using Nonnegative Matrix Factorization (NMF) and t-distributed Stochastic Neighbor Embeddings (TSNE).</li>
    <li>Generated predictions using Count and Term-Frequency Inverse-Document Frequency (TF-IDF) Vectorizers and Word2Vec embeddings, obtaining best genre recall and precision scores of up to 76%.</li>
  </ul>
</details>

### Predicting Sunspots Using Time Series with Prophet
Repository: [Sunspot-Timeseries](https://github.com/JoshuaGottlieb/Sunspot-Timeseries)

<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>A time series analysis project on sunspot data using Meta's Prophet modeling.</li>
    <li>Models were tuned and tested on yearly, monthly, and daily sunspot data to determine the best sampling frequency.</li>
    <li>Tuning improved the time series model’s performance from explaining less than 15% of the data variance up to explaining 61% of the data variance.</li>
  </ul>
</details>

### Predicting Customer Churn Using Scikit-Learn
Repository: [Telco-Churn-Predictions](https://github.com/JoshuaGottlieb/Telco-Churn-Predictions)

<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>Machine learning project designed to predict the underlying reasons behind customer churn at Telco.</li>
    <li>Exploratory Data Analysis was used to make initial predictions on feature importance, while Scikit-learn was used to test multiple models, including decision tree, random forest, XGBoosted tree, naive bayes, and logistic regression classifiers.</li>
    <li>The models were trained with and without SMOTE resampling to fix class imbalances, as well as with PCA transformations for dimensionality reduction. Each of the five models was hyperparameter tuned using GridSearchCV in the Scikit-learn package.</li>
    <li>Models were evaluated using the four main classification metrics: Accuracy, Precision, Recall, and F1-Score. The models were also evaluated for robustness using Receiver-Operating Characteristic (ROC) curves and Precision-Recall curves (PRC) to test for the tradeoffs in model performance at different thresholding values.</li>
    <li>The best model for the dataset was the Random Forest model trained on the SMOTE resampled data, resulting in an 81% Recall and 77% Accuracy, with an ROC AUC of 0.862.</li>
  </ul>
</details>

### Exoplanet Classification using NASA Data
Repository: [Exoplanet-Classification](https://github.com/JoshuaGottlieb/Exoplanet-Classification)

<details>
  <summary>Click for More Information</summary>
  <ul>
    <li>Classification project to predict the existence of exoplanets using trees, ensemble methods, and boosted tree algorithms.</li>
    <li>Obtained 9,500 exoplanet records using the NASA Exoplanet Archive API.</li>
    <li>Utilized scikit-learn Pipelines and GridSearchCV to automate preprocessing and model selection.</li>
    <li>Addressed class imbalances using imblearn's RandomOverSampler and SMOTE algorithms.</li>
    <li>Tested a variety of classifiers, including K-Nearest Neighbors, Random Forests, AdaBoost, and XGBoost, obtaining an 85.7% accuracy classifying exoplanets.</li>
  </ul>
</details>

## Website Scraping and Semantic Analysis

### Website Credibility Scorer for Semantic Relevance to User Query

Repository: [Chatbot-Credibility-Scorer](https://github.com/JoshuaGottlieb/Chatbot-Credibility-Scorer)

<details>
  <summary>Click for More Information</summary>
  
  <b>Under Deliverable 2:</b>
  
  <ul>
    <li>Created a scripting algorithm that evaluates the credibility of a given website URL based on the results of a user search through a search engine such as Google or SERP API. URLs are rated based on domain trustworthiness, URL title relevance, page content relevance, and Google Scholar citations. </li>
    <li>Pages were scraped using requests and BeautifulSoup, and outgoing links were automatically extracted for use in assessing outgoing credibility of cited sources, with irrelevant text blocks and advertisements expunged.</li>
    <li>Content and title relevances were calculated by using SentenceTransformer models to calculate semantic similarity between the page content and the search query.</li>
  </ul>

  <b>Under Deliverable 3:</b>
  
  <ul>
    <li>Pages are given a credibility score between 1 and 5 stars. To test the effectiveness of the credibility scorer, several hundred webpages were manually graded by 17 different people, and a neural network was trained to predict the human credibility score from the user search query and the machine-generated credibility score. The neural network utilized the all-MiniLM-L6-v2 Sentence transformer with binary quantization to embed the user search query into a 48 dimensional space, after which it was input through a 4 layer dense artificial neural network using ElasticNet regularization and an ExponentialDecay learning rate scheduler. </li>
    <li>The final Tensorflow model obtained a validation accuracy of 72.13% on a 5 class problem using only 300 samples, indicating a reasonably strong ability to predict the human credibility score from the user search and the machine-generated score, showing that the machine-generated scores were fairly close to the scores chosen by humans.</li>
  </ul>
</details>
