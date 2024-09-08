from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import List
import os
import base64
import requests
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import json
from openai import OpenAI

import mysql.connector

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Load environment variables
load_dotenv()

pinecone_index_name = os.getenv('INDEX_NAME')

model = ChatOpenAI(temperature=1, model="gpt-4o")
client = OpenAI()

class Item(BaseModel):
    title: str = Field(description="Title of the Dish")
    ingredients: List[str] = Field(description="Ingredients required for the recipe")
    recipe: List[str] = Field(description="Detailed recipe for the Dish")

class ItemsList(BaseModel):
    items: List[Item]

class RecipeItem(BaseModel):
    title: str = Field(description="Name of the Recipe")
    ingredients: List[str] = Field(description="Ingredients required for the recipe")
    instructions: List[str] = Field(description="Detailed step-by-step instructions for the recipe")
    tips: List[str] = Field(description="Suggested tips for the recipe")

class Recipe(BaseModel):
    items: List[RecipeItem]



def image_to_base64(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_data = response.content
        base64_encoded_data = base64.b64encode(image_data).decode('utf-8')
        print("Image downloaded and converted to base64.")
        return f"data:image/jpeg;base64,{base64_encoded_data}"
    else:
        raise Exception(f"Failed to download image from {image_url}")

def extract_ingredients(image_base64):
    prompt = "You are a meal planner bot. Your task is to analyze the image to recognize the list of ingredients in the image."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64,  
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content

def generate_recipe(ingredients: str, total_recipe):
    # Setup the parser
    parser = JsonOutputParser(pydantic_object=Recipe)

    # Define the format instructions
    format_instructions = parser.get_format_instructions()

    # Create the prompt template
    prompt = PromptTemplate(
        template="""
        Generate a list of unique and delicious recipe using the following set of ingredients: {ingredients}.

        Craft a step-by-step cooking guide, including instructions on preparation, cooking techniques, and any additional seasonings or garnishes.
        Be creative and provide tips for enhancing flavors. Consider dietary preferences or restrictions if specified.
        Feel free to suggest variations or substitutions to accommodate different tastes.

        provide with me with proper complete recipe


        \n\n
        give me a total of strictly {total_recipe} recipes (important)


        \n\n
        {format_instructions}
        """,

        input_variables=["ingredients","total_recipe"],
        partial_variables={"format_instructions": format_instructions},
    )

    # Create the chain (assuming `model` is an instance of a language model that you are using)
    recipe_chain = prompt | model | parser

    # Pass the ingredients to the LLM chain to generate a recipe
    recipe_response = recipe_chain.invoke({"ingredients": ingredients, "total_recipe":total_recipe})


    return recipe_response["items"]


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
    category_name = (data.get('category_name'))

    metadata_filter = {
        "_iky_recipes_makes": {"$gte": number_of_people},
        "_iky_recipes_cooking_time": {"$lte": time_for_cooking},
        # "category_name": {category_name}
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
                "video": doc.metadata.get("_iky_recipes_video"),
            }
            for doc, score in similarity_search_results
        ]
    else:
        response = []

    return response

# Function to generate items if vector search yields no results
def generate_items(data: dict, input: str, total_recipe):
    number_of_items = calculate_number_of_items(data)
    
    template = f"""
        You are a meal planner. You are given with user's meal preferences. Based on these preferences, you need to generate {number_of_items} meals for lunch and dinner.
        User's meal preferences: {input}
        
    """

    # Initialize the parser
    parser = JsonOutputParser(pydantic_object=ItemsList)

    # Create the prompt template
    prompt = PromptTemplate(
        template="Answer the query: {query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Chain the prompt, model, and parser
    chain = prompt | model | parser
    result = chain.invoke({"query": template})

    items = result["items"]
    return items

# Function to divide items into days
def divide_into_days(items, days: int):
    print(items)
    days_meal_plan = {}
    items_per_day = max(1, len(items) // days)  # Ensure at least one item per day if items exist

    for day in range(1, days + 1):
        start_index = (day - 1) * items_per_day
        end_index = start_index + items_per_day
        # Avoid out-of-bounds error if there are fewer items than needed
        days_meal_plan[f"Day {day}"] = items[start_index:end_index] if start_index < len(items) else []

    return days_meal_plan

@app.route('/')
def home():
    return "Welcome to the Meal Plan Generator API!"

@app.route('/generate-meal-plan', methods=['GET'])
def generate_meal_plan():
    # data = request.json
    data = {
        "meal_pln_duration": request.args.get('meal_pln_duration'),
        "time_for_cooking": request.args.get('time_for_cooking'),
        "number_of_people": request.args.get('number_of_people'),
        "Culinary_Preferences": request.args.get('Culinary_Preferences'),
        "budget_min": request.args.get('budget_min'),
        "budget_max": request.args.get('budget_max'),
        "Dietary_Restrictions": request.args.get('Dietary_Restrictions'),
        "uploaded_image": request.args.get("image_url","")
    }

    print(json.dumps(data, indent=4))

    # Check if uploaded image has data
    if data.get("uploaded_image", ""):
        # Extract ingredients from the image
        image_base64 = image_to_base64(data["uploaded_image"])
        ingredients = extract_ingredients(image_base64)

        # Generate a recipe using the extracted ingredients
        recipe = generate_recipe(ingredients, int(data.get('meal_pln_duration', 1))*2)

        response = {
            "source": "3",
            "meal_plan": recipe,
        }
    else:
        # Optimize input
        input = optimised_input(data)

        # Perform vector search
        items_list = vectorsearch(input, data)

        # Generate items if no results from vector search
        if not items_list:
            items_list = generate_items(data, input, int(data.get('meal_pln_duration', 1))*2)
            response = {"source":"1", "meal_plan":items_list}
        else:
            response = {"source":"2", "meal_plan":items_list}

        # Divide items into days
    days_meal_plan = divide_into_days(response["meal_plan"], int(data.get('meal_pln_duration', 1)))
    response["meal_plan"] = days_meal_plan

        # response = {"source": "1" if not items_list else "2", "meal_plan": days_meal_plan}

    print(response)
    # Return the response as JSON
    return response

@app.route('/save-meal-plan', methods=['POST'])
def save_meal_plan():
    # Connect to the database
    try:
        db = mysql.connector.connect(
            host="",
            port=,
            user="",
            password="",
            database=""
        )
    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)}), 500

    data = request.get_json()
    
    user_id = data.get('userID')
    meal_plan_name = data.get('mealPlanName')
    meal_plan = data.get('mealPlanData')

    cursor = db.cursor()

    try:
        # Insert the meal plan name and data into the MealPlans table
        cursor.execute("""
            INSERT INTO MealPlans (meal_plan_name, meal_plan_json)
            VALUES (%s, %s)
        """, (meal_plan_name, json.dumps(meal_plan)))
        
        meal_plan_id = cursor.lastrowid

        # Map the meal plan to the user in the UsersMealPlans table
        cursor.execute("INSERT INTO UsersMealPlans (user_id, meal_plan_id) VALUES (%s, %s)", (user_id, meal_plan_id))

        # Commit the transaction
        db.commit()

        return jsonify({"status": "success", "message": "Meal plan saved successfully!"})

    except mysql.connector.Error as err:
        db.rollback()
        return jsonify({"status": "error", "message": str(err)}), 500

    finally:
        cursor.close()
        db.close()

@app.route('/your-meal-plan', methods=['GET'])
def get_meal_plan():
    user_id = request.args.get('userID')

    # Connect to the database
    try:
        db = mysql.connector.connect(
            host="",
            port=,
            user="",
            password="",
            database=""
        )
    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)}), 500
    
    cursor = db.cursor(dictionary=True)

    try:
        # Retrieve meal plans associated with the user_id
        cursor.execute("""
            SELECT mp.id, mp.meal_plan_name, mp.meal_plan_json, mp.created_at
            FROM MealPlans mp
            JOIN UsersMealPlans ump ON mp.id = ump.meal_plan_id
            WHERE ump.user_id = %s
        """, (user_id,))
        
        meal_plans = cursor.fetchall()

        if not meal_plans:
            return jsonify({"status": "error", "message": "No meal plans found for the given user_id"}), 404

        for meal_plan in meal_plans:
            meal_plan['meal_plan_json'] = json.loads(meal_plan['meal_plan_json'])
        
        return jsonify({"status": "success", "meal_plans": meal_plans})

    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)}), 500

    finally:
        cursor.close()
        db.close()


@app.route('/get-meal-plan/<int:meal_plan_id>', methods=['GET'])
def get_meal_plan(meal_plan_id):
    # Connect to the database
    try:
        db = mysql.connector.connect(
            host="185.197.195.151",
            port=3306,
            user="external",
            password="/sK&?R,D5omh",
            database="iky"
        )
    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)}), 500

    cursor = db.cursor(dictionary=True)

    try:
        # Retrieve the meal plan based on meal_plan_id
        cursor.execute("SELECT * FROM MealPlans WHERE id = %s", (meal_plan_id,))
        meal_plan = cursor.fetchone()

        # Check if meal plan exists
        if not meal_plan:
            return jsonify({"status": "error", "message": "Meal plan not found"}), 404

        return jsonify({"status": "success", "meal_plan": meal_plan})

    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)}), 500

    finally:
        cursor.close()
        db.close()



@app.route('/delete-meal-plan/<int:meal_plan_id>', methods=['DELETE'])
def delete_meal_plan(meal_plan_id):
    # Connect to the database
    try:
        db = mysql.connector.connect(
            host="185.197.195.151",
            port=3306,
            user="external",
            password="/sK&?R,D5omh",
            database="iky"
        )
    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)}), 500

    cursor = db.cursor()

    try:
        # Check if the meal plan exists before deleting
        cursor.execute("SELECT * FROM MealPlans WHERE id = %s", (meal_plan_id,))
        meal_plan = cursor.fetchone()

        if not meal_plan:
            return jsonify({"status": "error", "message": "Meal plan not found"}), 404

        # Delete the meal plan from the database
        cursor.execute("DELETE FROM MealPlans WHERE id = %s", (meal_plan_id,))
        db.commit()

        return jsonify({"status": "success", "message": "Meal plan deleted successfully"})

    except mysql.connector.Error as err:
        return jsonify({"status": "error", "message": str(err)}), 500

    finally:
        cursor.close()
        db.close()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
