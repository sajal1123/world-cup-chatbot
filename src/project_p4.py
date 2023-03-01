# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2022
# Project Part 4
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# Before running code that makes use of Word2Vec, you will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations from Blackboard
#
# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info, see project statement for more details
# user_input: A string of arbitrary length
# Returns: name as string
def extract_user_info(user_input):
    name = re.search("((^|\s)([A-Z][A-Za-z.'&\-]*)\s+([A-Z][A-Za-z.'&\-]*\s+){0,2}([A-Z][A-Za-z.'&\-]*)(\s+|$))", user_input)
    if name:
        return name.group().strip()
    return ""


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string 
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: preprocessing, see project statement for more details
# Args:
#   user_input: A string of arbitrary length
# Returns: A string of arbitrary length
def preprocessing(user_input):
    modified_input = ""
    ip = get_tokens(user_input)
    processed_tokens = []
    for token in ip:
        if token not in string.punctuation:
            processed_tokens.append(token.lower())
    modified_input = ' '.join(processed_tokens)
    return modified_input


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None
    # [YOUR CODE HERE]
    tfidf_train = vectorizer.fit_transform(training_documents)
    return vectorizer, tfidf_train


# Function: vectorize_test, see project statement for more details
# vectorizer: A trained TFIDF vectorizer
# user_input: A string of arbitrary length
# Returns: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
#
# This function computes the TFIDF representation of the input string, using
# the provided TfidfVectorizer.
def vectorize_test(vectorizer, user_input):
    def vectorize_test(vectorizer, user_input):
        # Initialize the TfidfVectorizer model and document-term matrix
        tfidf_test = None

        # [YOUR CODE HERE]
        processed_input = preprocessing(user_input)
        tfidf_test = vectorizer.transform([processed_input])

        return tfidf_test


# Function: train_nb_model(training_documents, training_labels)
# training_data: A sparse TfIDF document-term matrix, dtype: scipy.sparse.csr.csr_matrix
# training_labels: A list of integers (0 or 1)
# Returns: A trained model
def train_nb_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    nb_model = GaussianNB()

    # Write your code here.  You will need to make use of the GaussianNB fit()
    # function.  You probably need to transfrom your data into a dense numpy array.
    # [YOUR CODE HERE]
    dense_arr = training_data.toarray()
    nb_model.fit(dense_arr, training_labels)

    return nb_model

# Function: get_model_prediction(nb_model, tfidf_test)
# nb_model: A trained GaussianNB model
# tfidf_test: A sparse TFIDF representation of the input string of shape (1, X), dtype: scipy.sparse.csr.csr_matrix
# Returns: A predicted label for the provided test data (int, 0 or 1)
def get_model_prediction(nb_model, tfidf_test):
    # Initialize the output label
    label = 0

    # Write your code here.  You will need to make use of the GaussianNB
    # predict() function. You probably need to transfrom your data into a dense numpy array.
    # [YOUR CODE HERE]
    label = nb_model.predict(tfidf_test.toarray())
    return label


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300,)

    # [YOUR CODE HERE]
    if token not in word2vec.keys():
        return word_vector
    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300,)

    # [YOUR CODE HERE]
    processed = preprocessing(user_input)
    tokens = get_tokens(processed)
    ctr = [len(tokens)]*(300)
    for token in tokens:
        embedding = np.add(w2v(word2vec, token), embedding)
    embedding = np.divide(embedding, ctr)
    return embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Three instantiated machine learning models
#
# This function instantiates the three imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    logistic = LogisticRegression()
    svm = LinearSVC()
    mlp = MLPClassifier()

    # [YOUR CODE HERE]

    return logistic, svm, mlp


# Function: train_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model(model, word2vec, training_documents, training_labels):
    # Write your code here:
    train_data = []
    for document in training_documents:
        # print(type(document))
        train_data.append(string2vec(word2vec, document))
    model.fit(train_data, training_labels)
    return model


# Function: test_model(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    # Write your code here:
    embedding = []
    for doc in test_documents:
        embedding.append(string2vec(word2vec, doc))
    predictions = model.predict(embedding)
    # print(predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    return precision, recall, f1, accuracy


# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    num_words = 0
    # [YOUR CODE HERE]
    tokens = nltk.tokenize.word_tokenize(user_input)
    filtered_tokens = []
    for token in tokens:
        if token not in string.punctuation:
            filtered_tokens.append(token)
    num_words = len(filtered_tokens)
    return num_words

# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    wps = 0.0
    # [YOUR CODE HERE]
    sent_tokens = nltk.tokenize.sent_tokenize(user_input)
    word_count = count_words(user_input)
    wps = word_count/len(sent_tokens)
    return wps


# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tagged_input = []
    # [YOUR CODE HERE]
    tokens = nltk.tokenize.word_tokenize(user_input)
    filtered = []
    for t in tokens:
        if t not in string.punctuation:
            filtered.append(t)
    pos_tags = nltk.pos_tag(filtered)
    for i in range(len(filtered)):
        tagged_input.append(pos_tags[i])
    # print("TAGGED INUT : ", tagged_input)
    return tagged_input


# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    # [YOUR CODE HERE]
    for word, pos in tagged_input:
        if pos[0:2] == 'PR' or pos[0:2] == 'WP':
            if pos == 'PRP':
                num_prp += 1
            num_pronouns += 1
        elif pos[0] == 'D':
            num_articles += 1
        elif pos == 'VBD' or pos == 'VBN':
            num_past += 1
        elif pos[0] == 'M':
            num_future += 1
        elif pos[0] == 'I':
            num_prep += 1
    # print(num_pronouns, num_prp, num_articles, num_past, num_future, num_prep)
    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    num_negations = 0
    # [YOUR CODE HERE]
    negations = ['no', 'not', 'never', 'n\'t']
    tokens = nltk.tokenize.word_tokenize(user_input)
    for token in tokens:
        if token in negations:
            num_negations += 1
    return num_negations


# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    # DO NOT change these values for autograder to work correctly
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20

    # [YOUR CODE HERE]
    if num_words > num_words_threshold:
        informative_correlates.append(psychological_correlates["num_words"])
    if wps > wps_threshold:
        informative_correlates.append(psychological_correlates["wps"])
    d = dict()
    d['num_pronouns'] = num_pronouns
    d['num_prp'] = num_prp
    d['num_articles'] = num_articles
    d['num_past'] = num_past
    d['num_future'] = num_future
    d['num_prep'] = num_prep
    d['num_negations'] = num_negations
    ds = dict(sorted(d.items(), key = lambda x : x[1], reverse=True))
    ctr = 0
    for k, v in ds.items():
        if len(informative_correlates)<3:
            informative_correlates.append(psychological_correlates[k])
        else:
            break
    return informative_correlates


# -------------------------- New in Project Part 4 --------------------------
# Function: welcome_state
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    # *** Replace the line below with your updated welcome message from Project Part 1 ***
    return "Welcome to the FIFA WORLD CUP 2022 chatbot!\n"


# Function: get_name_state
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that requests the user's name and then
# processes the user's response to extract the name.

# changed the function so that it returns the user's name. For further use in the chatbot
def get_name_state():
    # Request the user's name and accept a user response of arbitrary length
    user_input = input("What is your name?\nYou:")

    # Extract the user's name
    name = extract_user_info(user_input)

    # Show name and thank the user
    user_input = print(f"Thanks {name}!")

    return name


# Function: sentiment_analysis_state
# model: The trained classification model used for predicting sentiment (best one)
# word2vec: The word2vec dictionary
# first_time (bool): indicates whether the state is active for the first time. HINT: use this parameter to determine next state.
# Returns: A string indicating the next state
#
# This function implements a state that asks the user for input and predicts their sentiment
def sentiment_analysis_state(user_input, model, word2vec, first_time=0):
    # Check the user's current sentiment
    # user_input = input("What do you want to talk about today?\n")

    # Predict user's sentiment
    w2v_test = string2vec(word2vec, user_input)

    label = None
    label = model.predict(w2v_test.reshape(1, -1)) # Use this if you select one of the other models (swap mlp for svm, etc.)
    print(label)
    if first_time==0:
        if label == 0:
            return "Hmm, it seems like you're not too excited. Maybe talking about it will get you in the spirit!"
        elif label == 1:
            return "It sounds like you're all set for the greatest tournament in the world!"
        else:
            return "Hmm, that's weird.  My classifier predicted a value of: {0}".format(label)
    elif first_time==1:
        if label == 0:
            return "Doesn't sound like you're too confident about their chances!"
        elif label == 1:
            return "Sounds promising! I hope they do well."
        else:
            return "Hmm, that's weird.  My classifier predicted a value of: {0}".format(label)
    else:
        if label == 1:
            return "That is awesome! I can sense your excitement while talking about this."
        elif label == 0:
            return "You don't seem too enthusiastic about this. Let's move on."
        else:
            return "Hmm, that's weird.  My classifier predicted a value of: {0}".format(label)

    return ""


# Function: stylistic_analysis_state
# This function does not take any arguments
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative linguistic correlates to
# psychological status.
def stylistic_analysis_state(user_input, first_time=False):
    user_input = user_input[0]
    print("Thank you for your responses! Here's a short stylistic analysis of your answers during this interaction.")
    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)

    # Uncomment the code below to view your output from each individual function
    # print("num_words:\t{0}\nwps:\t{1}\npos_tags:\t{2}\nnum_pronouns:\t{3}\nnum_prp:\t{4}"
    #      "\nnum_articles:\t{5}\nnum_past:\t{6}\nnum_future:\t{7}\nnum_prep:\t{8}\nnum_negations:\t{9}".format(
    #    num_words, wps, pos_tags, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations))

    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    answer = ""
    answer += "Based on my stylistic analysis, I've identified the following psychological correlates in your response:"
    for correlate in informative_correlates:
        answer += "\n"
        answer += "- {0}".format(correlate)


    return answer


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user should be able to indicate that they would like to quit
# (in which case the state should be "quit"), redo the sentiment analysis
# ("sentiment_analysis"), or redo the stylistic analysis ("stylistic_analysis").
def check_next_state():
    next_state = False
    # [YOUR CODE HERE]
    do_next = 0
    next_state_input = input("Would you like to continue chatting?\n1) Yes!\n2) No thanks\nYou: ")
    if len(re.findall(".*([yY]es)|(YES)|(1).*", next_state_input)) > 0:
        next_state = True
    if next_state:
        kind = input("You : What kind of analysis would you like?\n1) Sentiment\n2) Stylistic\n3) Both\nYou: ")
        while do_next ==0:
            if len(re.findall(".*([sS]entiment)|(1).*", kind)) > 0:
                do_next = 1
            elif len(re.findall(".*([sS]tylistic)|(2).*", kind)) > 0:
                do_next = 2
            elif len(re.findall(".*([bB]oth)|(3).*", kind)) > 0:
                do_next = 3
            else:
                kind = input("Oops, looks like your input was not compatible. Please try again:\nWhat kind of analysis would you like?\n1) Sentiment\n2) Stylistic\n3) Both\nYou: ")
    return next_state, do_next


    # next_state_input = input("Awesome! What would you like to do next?\n1) Sentiment Analysis\n2) Stylistic Analysis\n3) Quit\n")
    # if len(re.findall(".*([sS]entiment [Aa]nalysis)|([sS]entiment)|(1).*", next_state_input)) > 0:
    #     next_state = 1
    # elif len(re.findall(".*([sS]tylistic [Aa]nalysis)|([sS]tylistic)|(2).*", next_state_input)) > 0:
    #     next_state = 2
    # elif len(re.findall(".*([qQ]uit)|(3).*", next_state_input)) > 0:
    #     next_state = 3
    # else:
    #     print("It seems that your input is incompatible. Please enter again\n")
    # return next_state

def initialize_player_team_data():
    teams = {
        'Argentina': 'Ah, the most in-form team in the world! They are strong favourites to win for sure. Coming into the World Cup off the back of a 36-match unbeaten streak isn\'t too shabby. They will be looking to add to their recent Copa America victory.',
        'Brazil': 'That\'s a solid pick! They\'re the most successful footballing nation in history and who would want to bet against them winning a record 6th title this year?',
        'England': 'Nice pick! They have a lot of young talent and their recent trend is encouraging- reaching the Semi-finals in 2018 and the final in EURO 2020 last year. They have as good a chance as anyone!',
        'France': 'The reigning champions! They have an extremely talented squad, it will be interesting to see if they can repeat the heriocs from 4 years ago.',
        'Belgium': 'One last dance for their golden generation, this one! It would surely be amazing to see them win an international trophy for the first time.',
        'Uruguay': 'Winners of the first ever world cup! They have a strong team and in-form players. They could go all the way!',
        'Germany': 'They have struggled in international tournaments lately, but a team with such rich legacy cannot be counted out!',
        'Spain': 'One of the most passionate footballing nations! They are always among the favorites.',
        'Portugal': 'A deeply talented squad with a recent major tournament win in EURO 2016, Portugal will make waves in this tournament.',
        'USA': 'The hosts of the next world cup! They will be looking to leave a mark on this tournament!',
        'Iran': 'One of the best teams in Asia, and they tend to punch above their weight in world cups!',
        'Wales': 'They are an amazing team to watch. Always punching above their weight and providing a story to remember!',
        'Switzerland': 'They are a smart footballing side. Despite being a small nation they leave a mark on international tournaments- like knocking France out of EURO 2020 last year!',
        'Mexico': 'They are an incredible team to watch, their supporters create such a mesmerising atmosphere in the games, no matter where they are!',
        'Qatar': 'The hosts! They will be looking to put on a show for their fans and make the country proud, there are very few better motivators than that!',
        'Netherlands': 'They are a team in form! After a stellar qualifying campaign they will be looking to stamp their authority on the world cup.',
        'Senegal': 'THE AFCON champions! They are a highly skilled team who will definitely be a tough team for any opponent!',
        'Poland': 'A potential dark horse! If they get off to a good start they have a good chance of reaching the knockout rounds!',
        'Denmark': 'Tactically, one of the most robust sides. They will look to build on their successful EURO2020 campaign where they reached the semi-finals.',
        'Japan': 'A highly innovative side that tries out new tactics! They are a really fun side to watch!',
        'Croatia': 'Finalists last time around! They will want to go one step further this year.',
        'South Korea': 'They will try to replicate the heroics from 2002!'
    }
    players = {
        'Messi': "Greatest footballer of all time! He will be looking to complete his trophy cabinet by adding the missing piece- the World Cup.",
        'Ronaldo': 'His last ever world cup. He will want to go out with a bang!',
        'De Bruyne': 'One of the best attacking midfielders in the world! He\'ll look to lead Belgium to glory',
        'Modric' : 'Player of the tournament last time around, it will be a joy to watch this timeless maestro in action.',
        'Pedri' : 'He is quickly becoming one of the best midfielders in the world. Unbelievable to think that he\'s only a teenager!',
        'Suarez' : 'A veteran of the game, one of the best strikers of all time. His world cup will not be quiet- that\'s for sure!',
        'Mbappe': 'One of the best players in the world, and the only teenager to score in the world cup final since Pele! He is one to watch out for.',
        'Neymar': 'He has not been very impactful at world cups, this is his chance to shine and take over the world!',
        'Mane': 'African player of the year and the talisman of Senegal. He is one to watch out for.',
        'Benzema': 'Winner of the Ballon d\'or, he will surely have an impact at the world cup.',
        'Lewandowski': 'One of the best striker in the world!',
        'Son': 'He\'s a prolific player and will try to lead his country to a successful campaign.',
        'Alves': 'The most decorated player in football history! He wouldn\'t mind adding a world cup tp his trophy collection!',
        'di Maria': 'He\'s a key player and will play a critical role in their journey.',
        'de Jong': 'A silky midfielder who will look to make his mark on the biggest stage.',
        'Nunez': 'An in-form striker who will follow in the footsteps of his predecessors and mark an era for his country!',
        'Vinicius': 'One of the best wingers in the world.',
        'Hazard': 'A great of the game! He has been unlucky with injuries lately, but a player like him is always a threat.',
        'Griezmann': 'He led France\'s attack to glory in 2018. More of the same please will be the request from fans!',
        'Depay': 'Talisman of his country\'s attack and their recent resurgence!',
        'Kane': 'Arguably the best striker in the world right now. It will be a joy to watch him!',
        'Eriksen': 'He will bounce back stronger than ever after an unfortunate health-related incident that cut his EURO 2020 campaign short.',
        'Martinez': 'He\'s a character alright! One of the best keepers in the world!',
        'Gavi': 'A sensation!',
        'Cancelo': 'One of the most creative fullbacks in the world!',
        'Lautaro': 'Among the best strikers at the world cup, I sense a lot of goals coming!'
    }
    questions = ["What is your first world cup memory?", "Are you a football fan? Tell me about the team you support!",
                 "The next world cup will be hosted by the USA. Will you plan to attend a few matches live?",
                 "Who do you think will be the best player in this year's tournament?",]
    return teams, players, questions

def comment_on_fav(text, group):
    target_team = ''
    for k in group.keys():
        if len(re.findall('.*'+k.lower()+'.*', text.lower())) > 0:
            target_team = k
            break
    if len(target_team)>0:
        return group[target_team]
    else:
        return "Wow, that is an interesting choice!"

# Function: run_chatbot
# model: A trained classification model
# word2vec: The pretrained Word2Vec dictionary (leave empty if not using word2vec based model)
# Returns: This function does not return any values
#
# This function implements the main chatbot system --- it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot():
    # [YOUR CODE HERE]
    teams, players, questions = initialize_player_team_data()
    welcome_state()
    name = get_name_state()
    excited = input("Are you excited for the upcoming FIFA World Cup?\nYou: ")
    sentiment_analysis_state(excited, svm, word2vec, 0)
    fav_team = input(f"Tell me {name}, which team are you supporting in this year's world cup?\nYou: ")
    comment_on_fav(fav_team, teams)
    fav_player = input("Who is your favorite player?\nYou: ")
    comment_on_fav(fav_player, players)
    confidence = input("How confident are you about your team's chances?\nYou: ")
    sentiment_analysis_state(confidence, svm, word2vec, 1)
    stylistic_analysis_state(fav_team+fav_player+confidence)
    next_state, do_next = check_next_state()
    while next_state:
        # print("do_next = ", do_next)
        if do_next == 3:
            fav_team = input(f"Tell me {name}, are there any other teams that you like?\nYou: ")
            comment_on_fav(fav_team, teams)
            fav_player = input("Who is your favorite player from their team?\nYou: ")
            comment_on_fav(fav_player, players)
            confidence = input("How confident are you about this team's chances?\nYou: ")
            sentiment_analysis_state(confidence, svm, word2vec, 1)
            stylistic_analysis_state(fav_team+fav_player+confidence)
        elif do_next == 2:
            answer = input(questions[np.random.randint(0, len(questions)-1)]+'\n')
            stylistic_analysis_state(answer)
        else:
            answer = input(questions[np.random.randint(0, len(questions)-1)]+'\n')
            sentiment_analysis_state(answer, svm, word2vec, 2)
        next_state, do_next = check_next_state()
    print("Thank you for using the chatbot. Enjoy the World Cup!")

    #function that takes in a message and gives response

def initialize():
    teams, players, questions = initialize_player_team_data()
    documents, labels = load_as_list("dataset.csv")
    word2vec = load_w2v(EMBEDDING_FILE)
    logistic, svm, mlp = instantiate_models()
    logistic = train_model(logistic, word2vec, documents, labels)
    svm = train_model(svm, word2vec, documents, labels)
    mlp = train_model(mlp, word2vec, documents, labels)
    return {"svm": svm, 
            "word2vec":word2vec,
             "teams": teams,
              "players": players,
               "questions": questions }

def reply(prompt, msg, model, answers):
    
    answers.append(msg)
    print("Answer stored in list:", answers)
    if " no " in prompt.lower():
        return ""
    if prompt[:7] == "Welcome":
        name = extract_user_info(msg)
        return f"Hi {name}!"
    elif prompt == "Are you excited for the upcoming FIFA World Cup?":
        wc_sentiment = sentiment_analysis_state(msg, model["svm"], model["word2vec"], 0)
        return wc_sentiment
    elif prompt and "Which team are you supporting in this year's world cup?" in prompt:
        fav_team = comment_on_fav(msg, model["teams"])
        return fav_team
    elif prompt == "Who is your favorite player?":
        fav_player = comment_on_fav(msg, model["players"])
        return fav_player
    elif prompt == "How confident are you about your team's chances?":
        analysis = sentiment_analysis_state(msg, model["svm"], model["word2vec"], 1)
        return analysis
    
    #test later
    elif prompt == "Would you like a stylistic analysis of your replies so far?":
        analysis = stylistic_analysis_state(answers, True)
        return analysis
    elif "Are there any other teams that you like?" in prompt:
        fav_team = comment_on_fav(msg, model["teams"])
        print(answers)
        return fav_team
    elif prompt == "Who is your favorite player from their team?":
        return comment_on_fav(msg, model["players"])
    elif prompt == "How confident are you about this team's chances?":
        analysis = sentiment_analysis_state(msg, model["svm"], model["word2vec"], 1)
        return analysis
        # stylistic_analysis_state(fav_team+fav_player+msg)
    # elif prompt == "Welcome to the FIFA World Cup Chatbot! What's your name?":
    #     pass
    elif prompt == "Would you like to continue?":
        return "You are fun to talk to!"
    else:
        return "Thank you for using the chatbot. Have a nice day!"



# ----------------------------------------------------------------------------




# Use this main function to test your code. Sample code is provided to assist with the assignment,
# feel free to change/remove it. In project components, this function might be graded, see rubric for details.
if __name__ == "__main__":

    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)

    # Instantiate and train the machine learning models
    logistic, svm, mlp = instantiate_models()
    logistic = train_model(logistic, word2vec, documents, labels)
    svm = train_model(svm, word2vec, documents, labels)
    mlp = train_model(mlp, word2vec, documents, labels)

    # Uncomment the line below to test out the w2v() function.  Make sure to try a few words that are unlikely to
    # exist in its dictionary (e.g., "covid") to see how it handles those.
    # print("Word2Vec embedding for {0}:\t{1}".format("vaccine", w2v(word2vec, "vaccine")))

    # Test the machine learning models to see how they perform on the small test set provided.
    # Write a classification report to a CSV file with this information.
    # Loading the dataset
    test_documents, test_labels = load_as_list("test.csv")
    models = [logistic, svm, mlp]
    model_names = ["Logistic Regression", "SVM", "Multilayer Perceptron"]
    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row
    i = 0
    while i < len(models): # Loop through other results
        p, r, f, a = test_model(models[i], word2vec, test_documents, test_labels)
        if models[i] == None: # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i],"N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i], p, r, f, a])
        i += 1
    outfile.close()

    # For reference, let us also compute the accuracy for the Naive Bayes model from Project Part 1
    # Fill in the code templates from your previous submission and uncomment the code below
    # vectorizer, tfidf_train = vectorize_train(documents)
    # lexicon = [preprocessing(d) for d in test_documents]
    # tfidf_test = vectorizer.transform(lexicon)
    # naive = train_nb_model(tfidf_train, labels)
    # predictions = naive.predict(tfidf_test.toarray())
    # acc = np.sum(np.array(test_labels) == predictions) / len(test_labels)
    # print("Naive Bayes Accuracy:", acc)

    # Reference code to run the chatbot
    # Replace MLP with your best performing model
    # run_chatbot(mlp, word2vec)


    #keep adding responses to list to return analysis

