# Cal Capstone Project: Detect Comm

_Module 20.1 Initial Report and Exploratory Data Analysis (EDA)_

_By Brad Brown_

## Executive summary

### Goal
Using my previous research project and paper (["Helping an LLM improve its detection of TV Commercials in Sports Broadcasts"](https://bb-wg02.github.io/cal-cap/scope/CS229_Stanford_Poster_-_Brad_Brown.pdf)) as a starting point, can I improve on my initial research to achieve higher accuracy and more granular **detection of commercials within a television sports broadcast?** 

The previous research project and paper was completed in March of 2024. The project had two stages. Stage 1 fed 15-second 'chunks' of text from 722 minutes of transcripts of the audio of TV sports broadcasts to ChatGPT via a python api, instructing it to categorize each chunk as a commercial or a sports broadcast. In Stage 2, those answers along with other statistics about each chunk were then used to train a logistic regression model to try to improve the accuracy of the initial predictions. 

The business goal of the Cal Capstone project is to improve the prediction accuracy overall. The Capstone project will be particularly focused on improving the Stage 2 accuracy: 
1) Rework stage 1 and stage 2 model processing and metrics to **predict whether each sentence is a commercial or not**
2) **Use feature engineering** to improve accuracy and quality of the stage 2 model
3) Use the **XGBoost model rather than logisitc regression model**

See the section *'How we measure success'* below for detailed specific objectives.


### How to get to the goal

1) We will understand and explore our data from stage 1 and its outputs. The outputs of stage 1 are the inputs to stage 2. Look for data quality errors and insights about potential feature engineering and prioritization hints.

2) We will decide a data split strategy to help us for feature engineering and the modeling process.

3) Will engineer features based on typical feature typical creation techniques and subject matter expertise. As an integral part of the modeling process we will try nonlinear variants of our input data and then use feature importance tools to narrow down to the most promising feature set.

4) With a set of features defined, we will tune the parameters of the XGBoost and the Logistic Regression models, searching for the best hyper-parameters of each to tune each model.

5) We will make a final model choice, tune it and then analyze how it performs on our held back test set.

6) We will then summarize our results, draw conclusions and define next steps


### Why does it matter

There are potentially millions of people that would benefit from the ability to seamless detect commercials while watching sporting events. Sporting events are very popular but commercials are the opposite - very few people desire to see them. If the transitions to and from commercials can be detected reliably, then the possibility of providing interesting content to consumers during commercials via non-TV devices is feasible. For example, short inspirational or educational snippets of video could be triggered on a consumer's phone during a commercial on TV device. Therefore, every improvement possible to make the detection reliable matters to those consumers. Current techniques use visual image and audio signal processing modeling techniques. This project represents one of the first times large language models and other ML modeling techniques on that LLM's output are used to solve the commercial detection problem. It has promise to be a more real-time practical solution or potentially be part of a more effective and efficient ensemble of modeling approach.

With the initial research project, F1 macro score of individual chunk predictions improved from Stage 1 (ChatGPT predictions) of 82% to 89% in Stage 2 (Logistic Regression predictions using stage 1 outcomes as input). A 7% improvement is good, however, 15-second chunk granularity reduces its potential for practical use. Consumers care about accuracy around these transition points from mainly primary content to mainly commercial content and vice versa. If detection of a chunk is wrong, it wastes 15-seconds of consumer attention. Wrong twice in a row and it wastes 30-seconds - this result could mean the entire solution gets rejected by end users.

Therefore, **this project moves to a more granular sentence-based approach rather than a '15-second chunk' based approach.** While sentence block prediction is a more challenging task, if successful, it makes it a more useful result, especially if by using feature engineering and XGBoost, the system can produce an end result that is still 89% F1 macro score or higher.

### Findings Summary
#### A straight forward 4% (F1 macro) overall improvement is possible in the Cal Capstone project approach. 

- Most of the overall 4% improvement is due to feature engineering.

#### An additional 5% improvement (total 9%) is possible in the Cal Capstone project approach but it is likely hard to achieve this in practice
- If the training data exposes the actual ground truth category of the previous sentence ('ground_truth_label_c_lag_1') and its relevant derived columns (e.g.,'num_blocks_sb_lag_1_gt') to the current sentence prediction, then the F1 score jumps to 98% from 93%.
- However, in a real-time system the corrected label feedback loop would not be that immediate - could not get the actual previous sentence category in under a second.

#### XGBoost vs Logistic Regression: XGB better
- XGBoost is 2.62% better

#### Sentence-based Prediction is much better than 15-second chunk-based prediction
- In Stage 1, the sentence-based approach is 5.80% more accurate. Perhaps more importantly, from a practical user experience perspective, sentence-based prediction is much better. The shorter duration(elapsed time) of a sentence means that an incorrect sentence wastes less end-user time than an error categorizing a 15-second chunk of text.



**How to measure success**


    We can measure success this way…


**Data Exploration and Insights**


    Data insights are …

**Feature Engineering**


    Features engineering process was …

**Model Development Process**


    The model development process was…

**Model Choice**


    We chose…

**Final Modeling Process**


    The final modeling process was …

**Outcomes**


    Our outcome was …

**Summary and Final Results**


    In summary the goal was ... and the final result was…

**Future Work**


    In the future we will…


----

*(Note: Full Github source code access to this Stanford Research Project available upon request, but to review the code, you can unzip a copy of the stanford project source code in this cal capstone github - see the 'previous_research' folder)*