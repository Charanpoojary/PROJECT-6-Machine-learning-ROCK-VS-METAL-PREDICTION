# ROCK-VS-METAL-PREDICTION-MACHINE-LEARNING-
we are building a system in Python that can predict whether an object is either Rock or Mine with SONAR Data. For this use case, we are using Logistic Regression Model for our prediction. 
Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@Charanpoojary 
Chaganti-Reddy
/
EVMarket-India
Public
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
EVMarket-India/README.md
@Chaganti-Reddy
Chaganti-Reddy Uploading project to github
Latest commit aedff7a on Jun 6
 History
 1 contributor
152 lines (104 sloc)  6.58 KB

Electric Vehicle Market Segmentation

Market segmentation becomes a crucial tool for evolving transportation technology such as electric vehicles (EVs) in emerging markets to explore and implement for extensive adoption. EVs adoption is expected to grow phenomenally in near future as low emission and low operating cost vehicle, and thus, it drives a considerable amount of forthcoming academic research curiosity. The main aim of this study is to explore and identify distinct sets of potential buyer segments for EVs based on psychographic, behavioral, and socio-economic characterization by employing an integrated research framework of ‘perceived benefits-attitude-intention’.
    
Table of Contents
Table of Contents
⚠️ Frameworks and Libraries
📖 Data Preprocessing
🔗 Download
🔑 Prerequisites
🚀  Installation
💡 How to Run
🔑 Results
👏 And it's done!
🙋 Citation
🔰 Future Goals
❤️ Owner
👀 License
⚠️ Frameworks and Libraries
SKLearn: Simple and efficient tools for predictive data analysis
Seaborn: Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
Plotly: The plotly Python library is an interactive, open-source plotting library that supports over 40 unique chart types covering a wide range of statistical, financial, geographic, scientific, and 3-dimensional use-cases.
KElbowVisualizer: The KElbowVisualizer implements the “elbow” method to help data scientists select the optimal number of clusters by fitting the model with a range of values for . If the line chart resembles an arm, then the “elbow” (the point of inflection on the curve) is a good indication that the underlying model fits best at that point. In the visualizer “elbow” will be annotated with a dashed line.
Matplotlib : Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
Numpy: Caffe-based Single Shot-Multibox Detector (SSD) model used to detect faces
Pandas: pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
📖 Data Preprocessing
Data pre-processing is an important step for the creation of a machine learning model. Initially, data may not be clean or in the required format for the model which can cause misleading outcomes. In pre-processing of data, we transform data into our required format. It is used to deal with noises, duplicates, and missing values of the dataset. Data pre-processing has the activities like importing datasets, splitting datasets, attribute scaling, etc. Preprocessing of data is required for improving the accuracy of the model.

🔗 Download
The dataset is now available here !

🔑 Prerequisites
All the dependencies and required libraries are included in the file requirements.txt See here

🚀  Installation
Clone the repo
$ git clone https://github.com/Chaganti-Reddy/EVMarket-India.git
Change your directory to the cloned repo
$ cd EVMarket-India
Now, run the following command in your Terminal/Command Prompt to install the libraries required
$ pip3 install -r requirements.txt

💡 How to Run
Open terminal. Go into the cloned project directory and type the following command:
$ python3 evmarket_india.py
🔑 Results
Correlation Matrix for the data set using Seaborn



Dendrogram for our data:



ScreePlot for our data:



Cluster analysis using silhouette:



Final Results:



Check out the report here

👏 And it's done!
Feel free to mail me for any doubts/query 📧 chagantivenkataramireddy1@gmail.com

🙋 Citation
You are allowed to cite any part of the code or our dataset. You can use it in your Research Work or Project. Remember to provide credit to the Maintainer Chaganti Reddy by mentioning a link to this repository and her GitHub Profile.

Follow this format:

Author's name - Chaganti Reddy
Date of publication or update in parentheses.
Title or description of document.
URL.
🔰 Future Goals
This study endeavoured to present EV taxonomy using an a-priori approach to categorize potential EV buyers into ub-segments of young and educated consumers and tested eticulously with a blended approach of multivariate and bivariate techniques

❤️ Owner
Made with ❤️  by Chaganti Reddy

👀 License
MIT © Chaganti Reddy

Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
