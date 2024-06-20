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


### Why it matters

There are potentially millions of people that would benefit from the ability to seamless detect commercials while watching sporting events. Sporting events are very popular but commercials are the opposite - very few people desire to see them. If the transitions to and from commercials can be detected reliably, then the possibility of providing interesting content to consumers during commercials via non-TV devices is feasible. For example, short inspirational or educational snippets of video could be triggered on a consumer's phone during a commercial on TV device. Therefore, every improvement possible to make the detection reliable matters to those consumers. Current techniques use visual image and audio signal processing modeling techniques. This project represents one of the first times large language models and other ML modeling techniques on that LLM's output are used to solve the commercial detection problem. It has promise to be a more real-time practical solution or potentially be part of a more effective and efficient ensemble of modeling approach.

With the initial research project, F1 macro score of individual chunk predictions improved from Stage 1 (ChatGPT predictions) of 82% to 89% in Stage 2 (Logistic Regression predictions using stage 1 outcomes as input). A 7% improvement is good, however, 15-second chunk granularity reduces its potential for practical use. Consumers care about accuracy around these transition points from mainly primary content to mainly commercial content and vice versa. If detection of a chunk is wrong, it wastes 15-seconds of consumer attention. Wrong twice in a row and it wastes 30-seconds - this result could mean the entire solution gets rejected by end users.

Therefore, **this project moves to a more granular sentence-based approach rather than a '15-second chunk' based approach.** While sentence block prediction is a more challenging task, if successful, it makes it a more useful result, especially if by using feature engineering and XGBoost, the system can produce an end result that is still 89% F1 macro score or higher.

### Findings Summary
#### A straight forward 4% (F1 macro) overall improvement is possible in the Cal Capstone project approach. 

- Most of the overall 4% improvement is due to feature engineering.

#### An additional 5% improvement (total 9%) is possible in the Cal Capstone project approach but it is likely hard to achieve this in practice
- If the training data exposes the actual ground truth category of the previous sentence and its relevant derived columns (e.g.,'Number of Sentence Since Sports Broadcast') to the current sentence prediction, then the F1 score jumps to 98% from 93%.
- However, in a real-time system the corrected label feedback loop would not be that immediate - could not get the actual previous sentence category in under a second.

#### XGBoost vs Logistic Regression: XGB better
- XGBoost is 2.62% better

#### Sentence-based Prediction is much better than 15-second chunk-based prediction
- In Stage 1, the sentence-based approach is 5.80% more accurate. Perhaps more importantly, from a practical user experience perspective, sentence-based prediction is much better. The shorter duration(elapsed time) of a sentence means that an incorrect sentence wastes less end-user time than an error categorizing a 15-second chunk of text.

![Model-Scores.png](https://bb-wg02.github.io/cal-cap/images/final_chart.png)



![act_vs_pred_comm_streak_new_stage2.png](https://bb-wg02.github.io/cal-cap/output/act_vs_pred_comm_streak_previous_stage2.png)

![act_vs_pred_comm_streak_new_stage2.png](https://bb-wg02.github.io/cal-cap/output/act_vs_pred_comm_streak_new_stage2.png)


#### Future Work and Development

#### Next Steps and Recommendations

#### How we measure success
The previous project correctly relied on Confusion Matrices as well as F1, Precision, and Recall accuracy metrics to judge commercial detection effectiveness. 
> The essence of the F1 Score lies in its ability to capture the trade-off between the two critical components of a model's performance: the precision, which measures how many of the items identified as positive are actually positive, and recall, which measures how many of the actual positives the model identifies. 
> From: [DeepGram](https://deepgram.com/ai-glossary/f1-score-machine-learning) , Feb 2024

The 'positive' in our case is a 'commercial detected'. This makes the F1 a useful measure for our case because our end-users want us to avoid sports broadcasts being labeled as commercials as well as not misclassify commercials as sports broadcasts. In other words, although precision is slightly more important, the recall also matters to our end-users, therefore, F1 captures these two metrics in one number. 

The choice of F1 *Macro average* rather than F1 *Micro average* is because the data population is NOT particularly imbalanced: The data suggests that generally there are roughly twice the number of sports broadcasting sentences compared to commercials. Since end-users value the minority class (commercial detection) somewhat more, a macro-averaged measure is more appropriate (Micro average works well with heavily imbalanced data sets).

The confusion matrix is an excellent mechanism to visualize the distribution of the data into 4 intuitive boxes. We can easily see how many true positive and true negatives there are, but also see determine whether or not our misclassifications tend towards false positives or tend towards false negatives.

In the Cal Capstone project, we will again use these measures, but there are changes being made to the data structures and our analysis of the business needs also inspire changes in our prediction approach: sentence-based and windowed predictions.

**Data structure change:Sentence-based:**
  
- If the project de-emphasized the contiguous 15-second chunks of text and instead relied on single sentences, would the Stage 1 CM and F1 accuracy worsen, stay the same, or improve?

**Stage 2 modeling changes: XGBoost and feature engineering:**
  
- Can we train an XGBoost Decision Tree model on the LLM answers and metadata about each sentence to improve the accuracy more than the previous use of a Logistic Regression model?
- Can we use feature engineering such as adding features that take the square of existing features to improve accuracy?
- If the model can rely on past history of correct labels (rather than only the history of Stage 1 estimates), will it help it predict current sentence label better?

**Lastly, but importantly, apply metrics for our results across different configurations to enable system designers to align more closely with the business goals and improve the user experience:**
 
1) **Windowed, rolling, or 'smoothed' predictions:** Adopting 'windowed' predictions in Stage 1, means three of the same prediction in a row are needed to change the current prediction. 
In a sentence-based model, will we experience  'flip flopping' where the incorrect categorizations of one sentence flips the user to the wrong context and then the next sentence flips them back and then the cycle repeats? If so, this cycle might be very annoying to end users. 
Are the errors that cause a flip-flop between corect and incorrect predictions sentence by sentence worse than a consistent set of incorrect predictions that are slower to transition to the correct answer after a shift to or from commercials? 
In essence is it better to have your errors grouped together within long chains of sentences? We will measure both ways. If each has close to the same accuracy, then the product designers can choose the experience that users prefer.

2) **Handling Embedded commercials:**
Should we differentiate between 'embedded commercials' vs 'standard commercials'. For example, oftentimes a sports broadcaster will promote a product while still talking about the sporting event in the same sentence: 'Let's take a look at our Coca Cola top player of the game statistics…'. *For this project, we will NOT attempt to categorize 'embedded commercials'.*

#### **In summary, the metrics we will use are F1 Macro, Recall and Precision as well as Confusion Matrices. They will be applied in mulitiple contexts.**
#### BASELINE:
##### -->A) Previous research baseline values
#### Cal Capstone:
##### -->B) Stage 1 using new sentence-based approach
#### For XGBoost model and Logistic Regression Models:
##### -->C) Stage 2 using new sentence-based approach WITHOUT feature engineering and WITHOUT Windowed Predictions
##### -->D) Stage 2 using new sentence-based approach WITH feature engineering but WITHOUT Windowed Predictions
##### -->E) Stage 2 using new sentence-based approach WITH feature engineering and WITH Windowed Predictions
##### -->F) Stage 2 like Stage E except we make the assumption that we know the actual (ground truth) category for the previous sentence
#####

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