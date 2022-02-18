# Usage: python3 app.py

from flask import Flask, render_template, request

import pandas as pd
import spacy
import random
import dateparser
import datetime 
from spacy.matcher import Matcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# amadeus sdk is used for getting real time information about prices of flight tickets and hotels
from amadeus import Client, ResponseError

amadeus = Client(
    client_id='XXXXXXX',
    client_secret='XXXXXX'
)

# Load traveling related intents corpus
intents_corpus = pd.read_csv("./travel_corpus.csv")

# Load and process Airport codes
fields = ['IATA','Name', 'City']
airport_df = pd.read_csv('./airports.csv', usecols=fields)
international_airports_df = airport_df[airport_df['Name'].apply(lambda x: 'International' in x)]

# Process corpus to extract responses for simple questions
simple_resp_df = pd.read_csv('./travel_corpus.csv')
simple_resp_df['response'] =  simple_resp_df['response'].apply(lambda x: x.strip('[]').replace("'","").split(', '))
simple_responses = dict(zip(simple_resp_df['label'], simple_resp_df['response'].tolist()))

# Load spaCy object
nlp = spacy.load('en_core_web_sm')


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_msg = request.args.get('msg')
    result = ""

    msg_intent = classify_user_intent(user_msg)

    doc = nlp(user_msg)

    if (msg_intent=='SearchFlight'):
        flight_request, err_resp = prepare_flight_request_data(doc)
        if (err_resp != ""):
            result = err_resp
        else:
            result = flight_response(flight_request)
    elif(msg_intent == 'SearchHotel'):
        hotel_request, err_resp = prepare_hotels_request_data(doc)
        if(err_resp != ""):
            result = err_resp
        else:
            result = hotels_response(hotel_request)
    else:
        result = random.choice(simple_responses[msg_intent])
     
    
    if len(result) != 0:
        return result
    else:
        return "Sorry, I can't help you with that! :("


def classify_user_intent(user_msg):
    '''
    Classifies user intent using tf-idf + cosine similarity approach
    '''
    vectorizer = TfidfVectorizer().fit(intents_corpus.text)
    train_corpus = vectorizer.transform(intents_corpus.text)

    def get_most_similar_sentence(cue, vectorizer,  train_corpus, top_n=3):
        # compute similarity to all sentences in the training corpus
        similarities = cosine_similarity(vectorizer.transform([cue]), train_corpus).flatten()
        related_docs_indices = similarities.argsort()[:-top_n:-1]

        most_relevant_idx = related_docs_indices[0]
        
        return (most_relevant_idx, similarities[most_relevant_idx], intents_corpus['text'].values[most_relevant_idx])
    
    (idx, similarity, text) = get_most_similar_sentence(user_msg, vectorizer, train_corpus)
    label = intents_corpus['label'].values[idx]
    
    return label

def create_matcher():
    matcher = Matcher(nlp.vocab)

    # Departure location
    pattern_start = [{'LOWER': 'from', },
                    {"ENT_TYPE": "GPE", "OP": "+"}]

    # Destination location
    pattern_end = [{'LOWER': 'to'},
                    {"ENT_TYPE": "GPE", "OP": "+"}]

    # Pattern IN
    pattern_in = [{'LOWER': 'in'},
                        {"ENT_TYPE": "GPE", "OP": "+"}]

    # Add patterns to matcher object
    matcher.add("START_LOC", [pattern_start])
    matcher.add("END_LOC", [pattern_end])
    matcher.add("IN_LOC", [pattern_in])
    return matcher

def match_locations(doc):
    #match_id, start and stop indexes of the matched words
    matcher = create_matcher()
    matches = matcher(doc)
    col_names = ['pattern', 'text', 'location']
    match_result = pd.DataFrame(columns=col_names)

    #Find all matched results and extract out the results
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  
        span = doc[start:end]  # The matched span
        loc = doc[start+1:end].text
        temp = pd.DataFrame([[string_id, span.text, loc]], columns=col_names)
        match_result = pd.concat([match_result, temp], ignore_index=True)
        
    return match_result

def named_entities_extraction(doc):
    '''
    takes an spacy doc object and makes a NER 
    and returns a dataframe of entities.
    '''
    col_names = ['text',  'label']
    sent_df = pd.DataFrame(columns=col_names)
    for ent in doc.ents:
        temp = pd.DataFrame([[ent.text, ent.label_]], columns=col_names)
        sent_df = pd.concat([sent_df, temp], ignore_index=True)
    return sent_df 

def flight_response(request):
    departure_city_code = request['depart_code']
    dest_city_code = request['destination_code']
    depart_date = request['depart_date']

    flight_resp = get_flight_tickets_response(departure_city_code, dest_city_code, depart_date)
    if(flight_resp == None or flight_resp.status_code != 200):
        resp = "Sorry, there was a problem fetching information about flight tickets price. :("
        return resp

    if (len(flight_resp.data) == 0):
        resp = 'No flights are available with the details you provided.'
        return resp
    
    cheapest_flight = flight_resp.data[0]
    priceObject = cheapest_flight['price']
    price = priceObject['total']+" "+priceObject['currency']

    depart_airport = international_airports_df[international_airports_df['IATA'] == departure_city_code]['Name'].iloc[0]
    dest_city = international_airports_df[international_airports_df['IATA'] == dest_city_code]['City'].iloc[0]

    resp = f'We have found the cheapest price for flight ticket from {depart_airport} to {dest_city} leaving on {depart_date}: {price}.'
    return resp 

def prepare_flight_request_data(doc):
    '''
    Extract all relevant entities and creates travel api request
    '''
    named_entities_df = named_entities_extraction(doc)

    # Get the number of dates and locations
    n_loc = len(named_entities_df[named_entities_df['label'] == 'GPE'])
    n_date = len(named_entities_df[named_entities_df['label'] == 'DATE'])

    # Return if there is not enough flight information.
    if (n_loc < 2 or n_date == 0):
        resp = "Sorry I don't understand. Please restate your flight request \
        with complete and valid locations names and departure date."
        return {}, resp

    matches_df = match_locations(doc)
    depart_code, destination_code = extract_depart_dest_airport_codes(matches_df)
    depart_date = extract_date(named_entities_df)
    
    return {"depart_code": depart_code, "destination_code":destination_code, "depart_date":depart_date}, ""

def extract_depart_dest_airport_codes(matches_df):
    departure_list = matches_df[matches_df['pattern']=='START_LOC'].loc[:,'location'].tolist()
    dest_list = matches_df[matches_df['pattern']=='END_LOC'].loc[:,'location'].tolist()
    
    departure_location = departure_list[-1]
    dest_loc = dest_list[-1]
    
    # Map location to an IATA airport code
    locs = international_airports_df[international_airports_df['City'] == departure_location]['IATA']
    depart_airport_code = international_airports_df[international_airports_df['City'] == departure_location]['IATA'].iloc[0]
    dest_code = international_airports_df[international_airports_df['City'] == dest_loc]['IATA'].iloc[0]
    
    return depart_airport_code, dest_code

def extract_date(named_entities_df):
    '''
    Extract date from the named_entities_df if there is such or return today date
    '''
    depart_date = None
    date_df = named_entities_df[named_entities_df['label'] == 'DATE']
    if (len(date_df) != 0):
        date_df = date_df.sort_values(by=['text'])
        depart_date = date_df['text'].iloc[0]
        depart_date = dateparser.parse(depart_date).strftime('%Y-%m-%d')
    else:
        depart_date = datetime.today().strftime('%Y-%m-%d')

    return depart_date

def get_flight_tickets_response(departure_airport_code, arrival_airport_code, depart_date):
    response = None
    try:
        response = amadeus.shopping.flight_offers_search.get(
        originLocationCode=departure_airport_code,
        destinationLocationCode=arrival_airport_code,
        departureDate=depart_date,
        adults=1,
        max=1)
    except ResponseError as error:
        print(error)

    return response

####### HOTELS ############
def hotels_response(city_code):
    hotel_resp = get_hotels_response(city_code)
    if (hotel_resp == None or hotel_resp.status_code != 200):
        resp = "Sorry, there was a problem fetching information about hotels. :("
        return resp

    if (len(hotel_resp.data) == 0):
        resp = 'No hotels are available with the details you provided.'
        return resp
    
    hotels_list = hotel_resp.data
    hotel_name = hotels_list[0]['hotel']['name']
    hotel_rating = hotels_list[0]['hotel']['rating']
    contact = hotels_list[0]['hotel']['contact']['phone']
    hotel_price = hotels_list[0]['offers'][0]['price']['total'] + " " + hotels_list[0]['offers'][0]['price']['currency']

    dest_city = international_airports_df[international_airports_df['IATA'] == city_code]['City'].iloc[0]

    resp = f'Our top suggestion for a very good available hotel with nice price in {dest_city} is {hotel_name}. It has {hotel_rating}-stars rating and the price for one night is {hotel_price}. If you are interested you can book directly on the following phone: {contact}.'
    return resp 

def prepare_hotels_request_data(doc):
    '''
    Extract all relevant entities and creates travel api request
    '''
    named_entities_df = named_entities_extraction(doc)

    n_loc = len(named_entities_df[named_entities_df['label'] == 'GPE'])

    if (n_loc != 1):
        err_resp = "Sorry I don't understand. Please restate your hotel request \
        with complete and valid city name."
        return {}, err_resp

    matches_df = match_locations(doc)
    city_code = extract_dest_city_code(matches_df)
    
    return city_code, ""

def extract_dest_city_code(matches_df):
    cities_list = matches_df[matches_df['pattern']=='IN_LOC'].loc[:,'location'].tolist()
    
    city_name = cities_list[-1]
    
    # Map city to an IATA airport code
    city_code = international_airports_df[international_airports_df['City'] == city_name]['IATA'].iloc[0]
    
    return city_code

def get_hotels_response(city_code):
    response = None
    try:
        # Get list of Hotels by city code
        response = amadeus.shopping.hotel_offers.get(cityCode=city_code)
    except ResponseError as error:
        print(error)

    return response

if __name__ == "__main__":
    app.run()