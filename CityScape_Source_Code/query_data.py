import warnings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

CHROMA_PATH = "chroma"
PREFERENCES_FILE = "user_preferences.json"
PLAN_FILE = "nyc_plan.json"

CLASSIFICATION_PROMPT = """
Analyze the following user query and classify it into the most appropriate category:

Query: {query}

Categories:
1. Food and Dining (restaurants, cafes, food-specific queries)
2. Entertainment (activities, shows, attractions)
3. Location/Navigation (finding places, directions)
4. Price/Budget Inquiry (costs, budget considerations)
5. Comparison (comparing venues or experiences)
6. Recommendations (seeking general suggestions)
7. Details/Information (specific venue information)
8. Generic Lunch/Dinner Suggestion (e.g. "Where can I grab a quick bite?","Recommend a restaurant to me","Where should I get lunch?","Where should I get dinner?")
9. Plan Management (e.g., "Add this to my plan", "Save this place", "Add it to the list") 

Also, determine if this is a preference statement (e.g., "I like Mexican food", "I prefer Italian restaurants").

Also, determine if it is a generic food question (e.g., "Recommend me a restaurant","Where can I grab a quick bite?","Suggest a place for lunch","Where should I get dinner?")

Provide your classification in this format:
Category: [main category]
Intent: [brief description of user intent]
Key Terms: [important terms from query]
Is Preference: [yes/no]
Preference Type: [cuisine/none]
Preference Value: [specific preference/none]
Is Generic Food Question: [yes/no]
"""

INITIAL_PROMPT_TEMPLATE = """
You are a knowledgeable NYC tour guide. Based on the query classification:
Category: {category}
Intent: {intent}

User Preferences: {preferences}

Using only the following venue information:
{context}

Question: {question}

Let's think about this step by step:

1. Understanding User Intent:
   - Primary goal: {intent}
   - Key terms to consider: {key_terms}
   - Consider user preferences: {preferences}
   - Any implicit requirements

2. Analyzing Available Options:
   - Match venues to user intent
   - Consider relevance to category: {category}
   - Evaluate ratings and reviews
   - Assess location and accessibility

3. Practical Considerations:
   - Time of day/week implications
   - Budget considerations
   - Special requirements
   - Local insights

Based on this analysis, provide a recommendation including:
- Name and exact location
- Why this matches their needs (including how it aligns with their preferences if applicable)
- Specific details about price, ratings, and features
- Practical tips for the best experience

Answer in a natural, helpful tone.
"""

FOLLOW_UP_PROMPT_TEMPLATE = """
Based on the previous recommendation and classification:
Category: {category}
Intent: {intent}

Using the previous venue information:
{context}

Previous Response:
{previous_response}

User Preferences: {preferences}

Follow-up Question: {question}

Provide a focused response that:
1. Specifically addresses the follow-up category ({category})
2. Maintains context from the previous recommendation
3. Gives precise, relevant details
4. Adds any helpful related information

Answer in a clear, direct manner.
"""

DAY_PLAN_PROMPT = """
Create a logical day itinerary using these venues:
{venues}

Consider:
1. Logical order based on location and type
2. Typical opening hours (assume standard business hours if not specified)
3. Typical time spent at each venue type
4. Travel time between venues
5. Meal times for restaurants
6. Energy levels throughout the day
7. Most efficient route through the city

Format the response as a detailed day plan with:
- Approximate timing for each venue
- Brief description of what to do there
- Logical transitions between places
- Travel suggestions between venues (subway/walking recommendations)
- Tips for making the most of each stop
- Suggested duration at each place

If the plan has mostly restaurants, space them out appropriately for meals and suggest complementary activities between them.
Include practical advice about the neighborhood and any insider tips for each stop.
"""

class NYCGuide:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        self.model = ChatOpenAI(temperature=0.7)
        self.last_response = None
        self.last_context = None
        self.last_classification = None
        self.plan = {"venues": []}
        self.load_preferences()
        self.load_plan()

    def load_preferences(self):
        try:
            with open(PREFERENCES_FILE, 'r') as f:
                self.preferences = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.preferences = {"cuisine": None}
            self.save_preferences()

    def save_preferences(self):
        with open(PREFERENCES_FILE, 'w') as f:
            json.dump(self.preferences, f)

    def load_plan(self):
        try:
            with open(PLAN_FILE, 'r') as f:
                loaded_plan = json.load(f)
                if isinstance(loaded_plan, dict) and "venues" in loaded_plan:
                    self.plan = loaded_plan
                else:
                    self.plan = {"venues": []}
        except (FileNotFoundError, json.JSONDecodeError):
            self.plan = {"venues": []}
            self.save_plan()

    def save_plan(self):
        with open(PLAN_FILE, 'w') as f:
            json.dump(self.plan, indent=2, fp=f)

    def update_preference(self, preference_type, value):
        self.preferences[preference_type] = value
        self.save_preferences()

    def extract_venue_info(self, context):
        extraction_prompt = f"""
        Extract the following information from this venue description:
        - Venue Name
        - Location
        - Type
        - Rating
        - Budget
        
        Text: {context}
        
        Return only the extracted information in this format:
        Name: [venue name]
        Location: [location]
        Type: [type]
        Rating: [rating]
        Budget: [budget]
        """
        
        response = self.model.invoke(extraction_prompt)
        info = {}
        for line in response.content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip().lower()] = value.strip()
        return info

    def add_to_plan(self):
        if not self.last_context:
            return "No venue to add to plan. Please ask about a place first."
        
        try:
            venue_info = self.extract_venue_info(self.last_context)
            
            if any(v.get('name', '').lower() == venue_info.get('name', '').lower() 
                  for v in self.plan.get('venues', [])):
                return f"{venue_info.get('name', 'This venue')} is already in your plan!"
            
            venue_entry = {
                **venue_info,
                'added_at': datetime.now().isoformat(),
                'context': self.last_context
            }
            
            if 'venues' not in self.plan:
                self.plan['venues'] = []
                
            self.plan['venues'].append(venue_entry)
            self.save_plan()
            
            return f"Added {venue_info.get('name', 'the venue')} to your NYC plan!"
            
        except Exception as e:
            print(f"Error adding to plan: {e}")
            return "Sorry, I couldn't add this place to your plan. Please try asking about the venue again."

    def get_plan_summary(self):
        if not self.plan.get('venues', []):
            return "Your NYC plan is empty! Ask me about places you'd like to visit."
        
        venues_for_summary = []
        for venue in self.plan.get('venues', []):
            venue_summary = {
                'name': venue.get('name', 'Unknown Venue'),
                'type': venue.get('type', 'Unknown Type'),
                'location': venue.get('location', 'Location not specified'),
                'budget': venue.get('budget', 'Budget not specified')
            }
            venues_for_summary.append(venue_summary)

        summary_prompt = f"""
        Create a summary of this NYC travel plan. For each venue, mention its key details.
        
        Venues:
        {json.dumps(venues_for_summary, indent=2)}
        
        Format the response as a clear plan with bullet points for each venue.
        Include the name, type, location, and budget for each venue.
        """
        
        try:
            response = self.model.invoke(summary_prompt)
            return response.content
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Sorry, I couldn't generate your plan summary. You have " + \
                   f"{len(self.plan.get('venues', []))} venues saved."

    def generate_day_plan(self):
        if not self.plan.get('venues', []):
            return "No venues saved in your plan!"
        
        venues_for_planning = []
        for venue in self.plan.get('venues', []):
            venue_summary = {
                'name': venue.get('name', 'Unknown Venue'),
                'type': venue.get('type', 'Unknown Type'),
                'location': venue.get('location', 'Location not specified'),
                'budget': venue.get('budget', 'Budget not specified'),
            }
            venues_for_planning.append(venue_summary)

        day_plan_prompt = ChatPromptTemplate.from_template(DAY_PLAN_PROMPT)
        prompt = day_plan_prompt.format(
            venues=json.dumps(venues_for_planning, indent=2)
        )

        try:
            response = self.model.invoke(prompt)
            return "\n🗽 Here's a suggested day plan with your saved places:\n\n" + response.content
        except Exception as e:
            print(f"Error generating day plan: {e}")
            return "Sorry, I couldn't generate a day plan with your saved venues."

    def classify_query(self, query):
        prompt_template = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        prompt = prompt_template.format(query=query)
        response = self.model.invoke(prompt)
        
        lines = response.content.strip().split('\n')
        classification = {
            'category': '',
            'intent': '',
            'key_terms': '',
            'is_preference': 'no',
            'preference_type': 'none',
            'preference_value': 'none',
            'is_generic_food_question': 'none'
        }
        
        for line in lines:
            if line.startswith('Category:'):
                classification['category'] = line.replace('Category:', '').strip()
            elif line.startswith('Intent:'):
                classification['intent'] = line.replace('Intent:', '').strip()
            elif line.startswith('Key Terms:'):
                classification['key_terms'] = line.replace('Key Terms:', '').strip()
            elif line.startswith('Is Preference:'):
                classification['is_preference'] = line.replace('Is Preference:', '').strip().lower()
            elif line.startswith('Preference Type:'):
                classification['preference_type'] = line.replace('Preference Type:', '').strip().lower()
            elif line.startswith('Preference Value:'):
                classification['preference_value'] = line.replace('Preference Value:', '').strip().lower()
            elif line.startswith('Is Generic Food Question:'):
                classification['is_generic_food_question'] = line.replace('Is Generic Food Question:', '').strip().lower()

        return classification

    def get_recommendations(self, query):
        if any(phrase in query.lower() for phrase in ['add to plan', 'add it to', 'save this', 'adding it']):
            return self.add_to_plan()

        classification = self.classify_query(query)
        self.last_classification = classification

        if classification['is_preference'] == 'yes' and classification['preference_type'] == 'cuisine':
            self.update_preference('cuisine', classification['preference_value'])
            return f"I've noted that you like {classification['preference_value']} food. I'll remember this for future recommendations!"

        enhanced_query = query
        if (classification['category'] in ['Food and Dining', 'Recommendations'] and 
            self.preferences['cuisine'] and classification['is_generic_food_question'] == 'yes'):
            enhanced_query = f"{query} {self.preferences['cuisine']} restaurant"

        results = self.db.similarity_search_with_relevance_scores(enhanced_query, k=4)
        if len(results) == 0 or results[0][1] < 0.7:
            return "I couldn't find any matching venues for that request. Could you try rephrasing or being more specific?"

        self.last_context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        preferences_to_pass = self.preferences if classification['category'] in ['Food and Dining', 'Recommendations'] and classification['is_generic_food_question'] == 'yes' else {"cuisine": None}

        prompt_template = ChatPromptTemplate.from_template(INITIAL_PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=self.last_context,
            question=query,
            category=classification['category'],
            intent=classification['intent'],
            key_terms=classification['key_terms'],
            preferences=json.dumps(preferences_to_pass)
        )

        response = self.model.invoke(prompt)
        self.last_response = response.content
        return response.content

    def handle_follow_up(self, question):
        if any(phrase in question.lower() for phrase in ['add to plan', 'add it to', 'save this', 'adding it']):
            return self.add_to_plan()

        if not self.last_response or not self.last_context:
            return "I don't have any previous recommendations to reference. Please ask about a specific place first."

        classification = self.classify_query(question)
        
        prompt_template = ChatPromptTemplate.from_template(FOLLOW_UP_PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=self.last_context,
            previous_response=self.last_response,
            question=question,
            category=classification['category'],
            intent=classification['intent'],
            preferences=json.dumps(self.preferences)
        )

        response = self.model.invoke(prompt)
        return response.content

    def is_follow_up_question(self, question):
        if any(phrase in question.lower() for phrase in ['add to plan', 'add it to', 'save this', 'adding it']):
            return True
            
        classification = self.classify_query(question)
        follow_up_categories = ['Location/Navigation', 'Price/Budget Inquiry', 'Details/Information']
        return classification['category'] in follow_up_categories and self.last_response is not None

def main():
    guide = NYCGuide()
    print("\n🗽 Welcome to NYC Explorer! Let's explore New York City today.")
    print("Tell me what you'd like to do or where you'd like to go!")
    print("You can:")
    print("- Tell me your preferences (e.g., 'I like Mexican food')")
    print("- Ask for recommendations")
    print("- Ask about specific places")
    print("- Ask follow-up questions")
    print("- Say 'add to plan' to save a place you like")
    print("\nType 'show plan' to see your saved places")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("\nYou: ").strip().lower()
        
        if user_input in ['quit', 'exit', 'bye', 'done']:
            if guide.plan.get('venues', []):
                print("\nHere's your final NYC plan:")
                print(guide.get_plan_summary())
                print("\nLet me organize these places into a day plan for you...")
                print(guide.generate_day_plan())
            print("\nThanks for exploring NYC! I'll remember your preferences and plan for next time! 👋")
            break
            
        if user_input == 'show plan':
            print("\nNYC Guide: " + guide.get_plan_summary() + "\n")
            continue
        
        if user_input == 'add to plan':
            response = guide.add_to_plan()
        elif guide.is_follow_up_question(user_input):
            response = guide.handle_follow_up(user_input)
        else:
            response = guide.get_recommendations(user_input)
            
        print("\nNYC Guide: " + response + "\n")

if __name__ == "__main__":
    main()