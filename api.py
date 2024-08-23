from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import List
import os
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import json 

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Load environment variables
load_dotenv()

pinecone_index_name = os.getenv('INDEX_NAME')

# Define the Pydantic model for your data structure
class Item(BaseModel):
    title: str = Field(description="Title of the Dish")
    ingredients: str = Field(description="Ingredients required for the recipe")
    recipe: str = Field(description="Detailed recipe for the Dish")

class ItemsList(BaseModel):
    items: List[Item]

# Function to calculate the number of items based on meal plan duration
def calculate_number_of_items(data: dict) -> int:
    meal_pln_duration = int(data.get("meal_pln_duration", "0").split('_')[0])
    return meal_pln_duration * 2

# Function to optimize the input for meal generation
def optimised_input(input: dict) -> str:
    meal_template = """
        You are given with a user's form data for generating a custom meal plan. You need to optimize the input JSON to create a plain text paragraph that is easily defined and easily understandable.
        User's form data :- {query}
    """
    prompt = PromptTemplate(
        template=meal_template,
        input_variables=["query"],
    )

    chain = prompt | model
    result = chain.invoke({"query": input})
    return result.content

# Function to perform a vector search
def vectorsearch(query: str, data: dict) -> str:
    number_of_items = calculate_number_of_items(data)
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings, text_key="text")

    number_of_people = int(data.get('number_of_people'))
    time_for_cooking = int(data.get('time_for_cooking').replace('-minutes', '').replace(' ', ''))

    metadata_filter = {
        "_iky_recipes_makes": {"$gte": number_of_people},
        "_iky_recipes_cooking_time": {"$lte": time_for_cooking}
    }

    similarity_search_results = vector_store.similarity_search_with_score(
        query=query,
        filter=metadata_filter,
        k=number_of_items,
    )

    scores = [score for _, score in similarity_search_results]
    average_score = sum(scores) / len(scores) if scores else 0

    if average_score > 0.7:
        response = [
            {
                "title": doc.metadata.get("post_title"),
                "url": doc.metadata.get("guid"),
                "image": doc.metadata.get("essb_cached_image"),
                # "source": "vectorstore"
            }
            for doc, score in similarity_search_results
        ]
    else:
        response = []

    return response

def update_keys(items):
    updated_items = []
    for item in items:
        updated_item = {
            'title': item.title,
            'ingredients': item.ingredients,  
            'recipe': item.recipe         
        }
        updated_items.append(updated_item)
    return updated_items

# # Function to generate items if vector search yields no results
# def generate_items(data: dict, input: str) -> List[Item]:
#     number_of_items = calculate_number_of_items(data)
#     example_output = """
#     Example Output Structure:
#     [
#         {
#             "title": "Vegetarian Tacos",
#             "ingredients": "Tortillas, Black beans, Corn, Bell peppers, Avocado, Salsa, Lime, Cilantro",
#             "recipe": "1. Heat tortillas. 2. In a pan, cook black beans and corn with bell peppers. 3. Assemble tacos with beans mixture, avocado slices, salsa, lime, and cilantro."
#         },
#         {
#             "title": "Vegetarian Enchiladas",
#             "ingredients": "Corn tortillas, Enchilada sauce, Black beans, Corn, Cheese, Onion, Bell peppers",
#             "recipe": "1. Preheat oven to 350°F. 2. Mix beans, corn, and bell peppers. 3. Fill tortillas with mixture, roll up, and place in baking dish. 4. Pour enchilada sauce over and top with cheese. 5. Bake for 20 minutes."
#         }
#     ]
#     """
#     template = f"""
#         You are a meal planner. You are given with user's meal preferences. You need to generate {number_of_items} meal plans based on these preferences.
#         User's meal preferences :- {input}    
        
#     """

#     parser = PydanticOutputParser(pydantic_object=ItemsList)

#     prompt = PromptTemplate(
#         template="{template}\n{format_instructions}",
#         input_variables=["template"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )

#     chain = prompt | model | parser
#     result = chain.invoke({"template":template})
#     items = result.items
#     # for item in items:
#     #     item.source = "gpt"

#     return items


# # Function to generate items if vector search yields no results
# def generate_items(data: dict, input: str) -> List[Item]:
#     number_of_items = calculate_number_of_items(data)
#     template = f"""
#         You are a meal planner. You are given with user's meal preferences. You need to generate {number_of_items} meal plans based on these preferences.
#         User's meal preferences :- {input}
#     """

#     parser = PydanticOutputParser(pydantic_object=ItemsList)

#     prompt = PromptTemplate(
#         template="{template}\n{format_instructions}",
#         input_variables=["template"],
#         partial_variables={"format_instructions": parser.get_format_instructions()},
#     )

#     chain = prompt | model | parser
#     result = chain.invoke({"template":template})
#     items = result.items

#     print("++++++++++++++++++++",result,"++++++++++++++++")

    # for item in items:
    #     item.source = "gpt"


    # return items
# Initialize the LLM model (e.g., using OpenAI's GPT-3 or GPT-4)




# Function to generate items if vector search yields no results
def generate_items(data: dict, input: str):
    number_of_items = calculate_number_of_items(data)
    

    data = {
        "items": [
            {
                "title": "here you need to give title of the dish",
                "ingredients": "here will be a list of ingredients that are required for this dish",
                "recipe": "here will be detailed recipe step by step instructions on how to cook"
            },
        ]
    }
    
    
    template = f"""
        You are a meal planner. You are given with user's meal preferences. You need to generate {number_of_items} meal plans based on these preferences.
        User's meal preferences: {input}

        Example Output Structure:
        {json.dumps(data, indent=4)}
        
        Remember, you just have to return an array of objects like the above example.
        You must always return perfect JSON.
    """

    # Initialize the parser
    parser = JsonOutputParser(pydantic_object=ItemsList)

    # Create the prompt template
    prompt = PromptTemplate(
        template="Answer the query: {query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Initialize the model
    # model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

    # Chain the prompt, model, and parser
    chain = prompt | model | parser
    result = chain.invoke({"query": template})

    print("++++++++++++++++++++", result, "++++++++++++++++")
    
    items = result["items"]
    return items





model = ChatOpenAI(temperature=0.5, model="gpt-4o")

@app.route('/')
def home():
    return "Welcome to the Meal Plan Generator API!"

@app.route('/generate-meal-plan', methods=['GET'])
def generate_meal_plan():
    # data = request.json

    data = {
        # "mode": request.args.get('mode'),
        "meal_pln_duration": request.args.get('meal_pln_duration'),
        "time_for_cooking": request.args.get('time_for_cooking'),
        "number_of_people": request.args.get('number_of_people'),
        "Culinary_Preferences": request.args.get('Culinary_Preferences'),
        "budget_min": request.args.get('budget_min'),
        "budget_max": request.args.get('budget_max'),
        "Dietary_Restrictions": request.args.get('Dietary_Restrictions')
    }

    print(json.dumps(data, indent=4))

    # Optimize input
    input = optimised_input(data)

    # Perform vector search
    items_list = vectorsearch(input, data)

    # Generate items if no results from vector search
    if not items_list:
        items_list = generate_items(data, input)
        # items_list = update_keys(items_list)
        response = {"source":"1", "meal_plan":items_list}
    else:
        response = {"source":"2", "meal_plan":items_list}

    print(response)
    # Return the response as JSON
    return response

















@app.route('/save-meal-plan', methods=['POST'])
def save_meal_plan():
    # Get the JSON data from the POST request
    data = request.get_json()

    # Print the received data in JSON format
    print("Received data:")
    # print()
    print(json.dumps(data, indent=4))

    # You can add logic here to save the data to a database or perform other operations

    # Return a success message
    return jsonify({"status": "success", "message": "Meal plan saved successfully!"})



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
