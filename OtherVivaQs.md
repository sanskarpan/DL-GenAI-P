## 
"1. Asked me to go through the GitHub repo. 
2. Asked whether I tried custom embeddings for my project or not. 
3. Suppose the language is French, will the specific models work for them? If yes then why? If not then how would you tackle the problem. 
4. Asked about salient features of all of my models. 
5. Asked about no_grad
6. Asked about gradient clipping
7. Asked how and why did I use only that model. 
8. Asked me about a few questions of my data preprocessing.
10. Asked how dropout works both in training and the inference part and how it'll still behave properly even when the neurons have learnt in each other's both absence and presence. "



## 
"
Show you ID, GitHub, WandB logs, Project report.

Show your notebook.

Show me your report and tell me overall what you have done.

How many models did you train?

What preprocessing did you do?

What did you use this particular preprocessing?

Explain the Tokenization used in all models ( WHY these choices)

What is BCEWithLogitsLoss ( i had used this)



Go to your custom/ scratch NN. Show the code.
-What is its design?
- how many hidden layers it has?
- Why did you chose this design?
- you used no_grad- why? 
- what is zero_grad, then?
- where is the gradient descent happening in this code?
- parameters of this model.



Go to your RNN model:
-- in depth questions about each line of the code.( many questions, couldn't answer)
-- difference between vanilla RNN & LSTM
-- why did you use gradient clipping?
-- weight decay

Go to the transformer model:
--- why did you use dataset again when you were already given a dataset(.csv)
--- what does Dataloader do?
--- what is masking?
--- Again, question for each line of code.


CODING:

design a custom, simple Neural network.( He gave the parameters)




"



##
"1) Checked my github commit history and quickly glanced through my report
2) Tokenizers I used
3) Why did I not use RNN instead of LSTM 
4) Which is better RNN or LSTM (RNN suffers from explosive gradients)
5) I used glove embeddings so he asked why I was training the weights again if im using glove
6) I used Roberta with Mean Pooling and in one line I had used tf.clip_by_value(sum_mask, 1e-9, 1e9) for my mask so he asked what was the use of this
7) If i had coded the same model with pytorch, will the output be different? Why?"





##
"Open and show your GitHub repo
Explain your file structure 
Checked for 3 weeks worth of commits
Open your report. What was the project about? 
Walk me through your report sections on preprocessing and models in brief. 
Explain the models with the code. 
Code specific questions? Why did you do what you have done? 
Why this particular architecture? 
What is this part of architecture doing? 
Why are you clipping gradients?(For overflow n normalisation apparently)
Parameter settings? Warm-up? Why? 
Did you try ensemble? 
Difference between RNN and LSTM? 
Code a scratch model acc to specifications given? 
Deployment? If yes please run it. If no what errors and what debugging steps have you taken? "



##
"1. Asked me to go through the tokenization strategy and models sections of the report quickly.

Then we shifted to notebooks. My first model was BiLSTM with a 3 layer classifier head.
2. Difference between LSTMs and RNNs.
3. What does nn.Embedding do? Why nn.Embedding and not others?
4. How does the embedding size affect the total model params? (Answer - Apart from the increase in the RNN units and hence params, nn.Embedding in itself contains weights to convert one hot encodings to dense vectors which will also get increased)
5. In which case do you think nn.Embedding will fetch you better results than any other pretrained tokenizer embeddings? (Answer - In the case where the dataset has no intersection with the text corpus the tokenizer is pre-trained on)
6. In the training loop, which line of code is responsible for applying gradient descent? Then what does loss.backward() do?
7. What is torch.nn.utils.clip_grad_norm_()?

My second model was Frozen BERT + a deep neural network.
8. Why this model?
9. What if you had unfrozen the BERT encoder layers? (This was an open ended question)
10. What is BatchNorm (used it in my classifier head) & Dropout? Why did you use them?

Didn't ask questions regarding my last model.
11. If I add a few more instances to the dataset but belonging to languages other than English such as Bengali, French etc., what would your approach be in training and classifying emotions this time?

He then asked me to show the wandb logs, and said ok.

Did not ask me to code or make any changes (prolly because he was late by 25 mins or so). His advice for L2:
1. Be prepared to write code.
2. Be prepared for situation based questions. 
3. Be thorough with the theory."






##
"Proctor is late my 17 min so not asked much questions 
1. Id card
2. Go to your report explain model artitecture 
3. Rnn vs lstm
3. Go to kaggle explain your notebook code
I explain well till by scratch model BiLSTM but i have not knowledge of transformer model code it is all from gpt and i have no time to practice it but my theory part is good for transformer 
4. Hello is only checking main class for all three model but in my transformer i implemented it through one block of functions for both bert and Roberta so he is not good for it
5. Is bert is encoder or decoder 
5. One question that stuck me ""if bert is encoder so it will generate embeddings how it is used in your muti classification""
6. Coding ""implement an simple binary classification ann which have 4 hidden layer including output""
7. What activation function used for output for this ann
8. Feedback from proctor ""i only acting i am listening i don't know what he is telling"""







##

"github history kaggle versions wandb logs
explain entire report
initial questions come from report. can ask any random question like why used ...
what is spectrogram then waht is mel spectrogram
for simple cnn model what is relu, maxpool2d, avgpool, conv2d, linear, batchnorm
what is the classifier in the model
explain training code
why adam optimizer
where is the gradient calculated
write code"





##
"1. Show github commits and their distribution
2. Asked to show wandb logs
3. explain all three models from report
4. show all three models in kaggle
5. what is 2d in conv2d, maxpool2d, etc for?
6. what does ReLU do?
7. What does MaxPool2d do?
8. Did you freeze any layers or tuned the entire pretrained model?
9. Why did you use efficientNetB2?
10. Difference between RNN and LSTM
11. code feedforward neural network with given params - can use documentation but no examples or AI
some other minor questions around models while explaining report"


