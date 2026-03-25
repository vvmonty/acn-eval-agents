import os
import re
import json
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict
from fractions import Fraction
from langfuse import observe

# Import your agent framework
from agents import Agent, function_tool

from src.utils.tools.gemini_grounding import google_search_grounded_sync

# Path to the recipe dataset
RECIPE_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/recipes.csv')

# --- 1. STRICT PYDANTIC MODELS ---

class Nutrients(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sodium: Optional[float] = Field(None, description="Sodium limit in mg")
    calories: Optional[float] = Field(None, description="Calorie limit (kcal)")
    fat: Optional[float] = Field(None, description="Total fat limit in grams")
    protein: Optional[float] = Field(None, description="Protein target in grams")
    carbohydrates: Optional[float] = Field(None, description="Carbohydrate limit in grams")

class RecipeParams(BaseModel):
    model_config = ConfigDict(extra='forbid') 
    max_total_time: int = Field(..., description="Maximum total cooking time in minutes")
    dietary_restrictions: List[str] = Field(default_factory=list)
    constraints: Optional[Nutrients] = Field(default=None)
    recipe_type: str = Field(..., description="Type of recipe (main dish, snack, side, drink)")

# --- 2. OPERATIONAL TOOLS ---
@observe()
def fetch_local_recipe(recipe_criteria_json: str) -> str:
    """Queries the local CSV for recipes. Returns 'NO_MATCH' if none satisfy criteria.

    recipe_criteria_json: JSON object with required keys max_total_time (int),
    recipe_type (str); optional dietary_restrictions (list of str) and
    constraints (object with optional numeric sodium, calories, fat, protein, carbohydrates).
    """
    try:
        params = RecipeParams.model_validate_json(recipe_criteria_json)
    except Exception as exc:
        return f"ERROR: Invalid recipe_criteria_json: {exc}"

    if not os.path.exists(RECIPE_DATA_PATH):
        return "NO_MATCH: Local database file missing. Search the web instead."
    
    try:
        df = pd.read_csv(RECIPE_DATA_PATH)
    except Exception:
        return "NO_MATCH: Could not read local database. Search the web instead."
    
    DIET_MAP = {
        "Vegetarian": ["chicken", "beef", "pork", "fish", "lamb", "shrimp", "seafood"],
        "Vegan": ["meat", "chicken", "beef", "pork", "fish", "egg", "milk", "cheese", "butter", "honey"],
        "Gluten-free": ["flour", "wheat", "barley", "rye"],
        "Dairy-free": ["milk", "cheese", "butter", "cream", "yogurt"],
        "Nut-free": ["peanut", "walnut", "almond", "cashew", "nut"],
        "Egg-free": ["egg", "mayonnaise"],
        "Low-sodium": ["salt", "soy sauce", "bouillon"],
        "Sugar-free": ["sugar", "syrup", "honey"]
    }

    # 1. Filter by Dietary Restrictions
    for restriction in params.dietary_restrictions:
        excluded = DIET_MAP.get(restriction, [])
        if excluded:
            df = df[~df['ingredients'].str.contains('|'.join(excluded), case=False, na=False)]

    # 2. Time Parsing
    def parse_time(t):
        if pd.isna(t): return 9999
        m = re.findall(r'(\d+)\s*(day|hr|min)', str(t).lower())
        total = 0
        for val, unit in m:
            if 'day' in unit: total += int(val) * 1440
            elif 'hr' in unit: total += int(val) * 60
            else: total += int(val)
        return total if total > 0 else (int(t) if str(t).isdigit() else 9999)

    df['total_mins'] = df['total_time'].apply(parse_time)
    results = df[df['total_mins'] <= params.max_total_time].copy()

    # 3. Filter by Recipe Type
    if params.recipe_type:
        results = results[results['cuisine_path'].str.contains(params.recipe_type, case=False, na=False)]

    # 4. Nutrition Filtering
    def get_nutr(nut_str, key):
        if not isinstance(nut_str, str): return 0
        match = re.search(rf"{key}\D*(\d+\.?\d*)", nut_str, re.IGNORECASE)
        return float(match.group(1)) if match else 0

    if params.constraints:
        active_limits = params.constraints.model_dump(exclude_none=True)
        for nutrient, limit in active_limits.items():
            results = results[results['nutrition'].apply(lambda x: get_nutr(x, nutrient) <= limit)]

    # 4. Rating Guardrail
    results['rating'] = pd.to_numeric(results['rating'], errors='coerce').fillna(0)
    results = results[results['rating'] >= 3.5]

    if results.empty:
        # Crucial: This string triggers the LLM fallback logic
        return "NO_MATCH: No local recipes meet criteria. ACTION REQUIRED: Use search_web to find an alternative."
    
    return results.sort_values('rating', ascending=False).head(1).to_json(orient='records')

@observe()
def check_cfia_recalls(ingredients_json: str) -> str:
    """SAFETY CHECK: Verifies ingredients against CFIA Food Recalls."""
    try:
        ingredients = json.loads(ingredients_json)
        recalls = [i for i in ingredients if "romaine" in i.lower()]
        return f"FAIL: Active recall on {recalls}" if recalls else "PASS"
    except: return "ERROR: Invalid ingredients format."


@observe()
def modify_recipe(recipe_json: str, target_servings: int, health_goal: str = "") -> str:
    """QUANTITY ENGINE: Scales all ingredients mathematically for target servings."""
    try:
        data = json.loads(recipe_json)
        recipe = data[0] if isinstance(data, list) else data
        
        orig_servings = int(recipe.get('servings', 1))
        scale_factor = target_servings / orig_servings
        
        def scale_quantity(match):
            val = match.group(0)
            try:
                if "/" in val:
                    return str(round(float(Fraction(val)) * scale_factor, 2))
                return str(round(float(val) * scale_factor, 2))
            except: return val

        updated_ingredients = []
        for ing in recipe['ingredients']:
            scaled_ing = re.sub(r'(\d+\/\d+|\d+\.\d+|\d+)', scale_quantity, ing)
            ing_low = ing.lower()
            if "salt" in ing_low:
                scaled_ing += " [ADAPTATION: Reduce per Health Canada]"
            elif any(f in ing_low for f in ["butter", "lard"]):
                scaled_ing += " [ADAPTATION: Swap for plant oils]"
            updated_ingredients.append(scaled_ing)

        recipe['ingredients'] = updated_ingredients
        recipe['servings'] = target_servings
        return json.dumps(recipe)
    except Exception as e:
        return f"ERROR in modify_recipe: {str(e)}"


@observe()
def get_local_recipe_type() -> List[Any]:
    """
    Extracts unique recipe types from the local dataset based on 'cuisine_path'.
    Returns:
        A sorted list of unique recipe types (e.g., main dish, snack, side, drink).
    """
    df = pd.read_csv(RECIPE_DATA_PATH)

    # Extract the 'cuisine_path' column
    cuisine_path_column = df['cuisine_path']

    # Split the values in the 'cuisine_path' column by a delimiter (e.g., '/')
    # Extract the first component of each path and collect unique values
    first_components = cuisine_path_column.dropna().apply(lambda x: x.split('/')[1]).unique()

    # Convert the result to a sorted list for better readability
    unique_first_components = sorted(first_components)

    return unique_first_components


@observe()
def prepare_shopping_list(recipe_json: str) -> str:
    """
    Generates an organized shopping list from modified recipe ingredients.
    Groups items by category and adjusts quantities to match realistic store minimums.
    
    Args:
        recipe_json: JSON string containing recipe with ingredients list
    
    Returns:
        JSON string with categorized shopping list, rounded to store units
    """
    try:
        data = json.loads(recipe_json)
        recipe = data[0] if isinstance(data, list) else data
        ingredients = recipe.get('ingredients', [])
        
        if not ingredients:
            return json.dumps({"error": "No ingredients found in recipe"})
        
        # Food categories
        CATEGORIES = {
            'Produce': ['tomato', 'onion', 'garlic', 'lettuce', 'carrot', 'celery', 'spinach', 'broccoli', 
                       'pepper', 'cucumber', 'potato', 'apple', 'banana', 'lemon', 'lime', 'berry', 'herb'],
            'Proteins': ['chicken', 'beef', 'pork', 'fish', 'salmon', 'shrimp', 'tofu', 'egg', 'meat', 'protein'],
            'Dairy': ['milk', 'cheese', 'butter', 'yogurt', 'cream', 'sour cream'],
            'Pantry': ['flour', 'sugar', 'salt', 'oil', 'sauce', 'spice', 'rice', 'pasta', 'bean', 'canned',
                      'vinegar', 'broth', 'stock', 'honey', 'peanut butter', 'cocoa'],
            'Frozen': ['frozen', 'ice'],
            'Beverages': ['juice', 'wine', 'beer', 'coffee', 'tea']
        }
        
        # Store minimums: defines realistic minimum quantities for items
        STORE_MINIMUMS = {
            'whole': {
                'tomato': 1, 'onion': 1, 'lemon': 1, 'lime': 1, 'apple': 1, 'banana': 1, 'egg': 6,
                'potato': 1, 'garlic': 1, 'pepper': 1, 'cucumber': 1, 'head': 1
            },
            'volume': {  # cups/tbsp -> minimum common store unit
                'cup': 0.5, 'tbsp': 1, 'tsp': 1, 'ml': 250, 'l': 1
            },
            'weight': {  # grams/lbs -> minimum package sizes (in original unit)
                'g': 100, 'oz': 4, 'lb': 1, 'kg': 0.5
            }
        }
        
        def normalize_quantity(qty_str, unit, ing_name):
            """
            Convert quantity to store-buyable amount.
            Returns (adjusted_qty, adjusted_unit, note)
            """
            if qty_str == "To taste":
                return qty_str, unit, ""
            
            try:
                # Parse fraction or decimal
                if "/" in qty_str:
                    qty_float = float(Fraction(qty_str))
                else:
                    qty_float = float(qty_str)
            except:
                return qty_str, unit, ""
            
            unit_lower = unit.lower().strip() if unit else ""
            ing_lower = ing_name.lower()
            
            # Check if it's a countable item (tomatoes, eggs, etc.)
            for item_name, min_qty in STORE_MINIMUMS['whole'].items():
                if item_name in ing_lower:
                    # Round up to minimum
                    adjusted = max(int(qty_float) if qty_float == int(qty_float) else qty_float, min_qty)
                    if adjusted > qty_float:
                        return str(int(adjusted)), "whole" if adjusted == int(adjusted) else unit, \
                               f"(rounded up from {qty_str} to store minimum)"
                    return qty_str, unit, ""
            
            # Check volume measurements
            if any(u in unit_lower for u in ['cup', 'tbsp', 'tsp', 'ml', 'l']):
                for u, min_vol in STORE_MINIMUMS['volume'].items():
                    if u in unit_lower:
                        if qty_float < min_vol:
                            return str(min_vol), unit, f"(rounded up from {qty_str} to store minimum)"
                        return qty_str, unit, ""
            
            # Check weight measurements
            if any(u in unit_lower for u in ['g', 'oz', 'lb', 'kg']):
                for u, min_wt in STORE_MINIMUMS['weight'].items():
                    if u in unit_lower:
                        if qty_float < min_wt:
                            return str(min_wt), unit, f"(rounded up from {qty_str} to store minimum)"
                        return qty_str, unit, ""
            
            return qty_str, unit, ""
        
        # Parse and categorize ingredients
        shopping_dict = {cat: {} for cat in CATEGORIES.keys()}
        shopping_dict['Other'] = {}
        
        for ing in ingredients:
            # Remove adaptation notes
            clean_ing = re.sub(r'\[ADAPTATION:.*?\]', '', ing).strip()
            
            # Extract quantity (e.g., "2 cups", "1.5 tbsp", "3/4 cup")
            qty_match = re.match(r'^([\d.]+(?:/\d+)?)\s*([a-z]+)?', clean_ing, re.IGNORECASE)
            quantity = qty_match.group(1) if qty_match else "To taste"
            unit = qty_match.group(2) if qty_match and qty_match.group(2) else ""
            
            # Remove quantity from ingredient name
            ing_name = re.sub(r'^[\d.]+(?:/\d+)?\s*[a-z]*\s*', '', clean_ing, flags=re.IGNORECASE).strip()
            
            # Normalize quantity to store minimum
            adjusted_qty, adjusted_unit, norm_note = normalize_quantity(quantity, unit, ing_name)
            
            # Categorize
            category = 'Other'
            for cat, keywords in CATEGORIES.items():
                if any(kw in ing_name.lower() for kw in keywords):
                    category = cat
                    break
            
            # Consolidate quantities (simple addition for same items)
            if ing_name not in shopping_dict[category]:
                shopping_dict[category][ing_name] = {
                    'original_quantity': quantity,
                    'original_unit': unit,
                    'quantity': adjusted_qty,
                    'unit': adjusted_unit,
                    'note': norm_note,
                    'raw': ing
                }
        
        # Format into readable shopping list
        result = {
            'recipe_name': recipe.get('recipe_name', 'Unknown Recipe'),
            'total_servings': recipe.get('servings', 1),
            'shopping_list': shopping_dict
        }
        
        # Create formatted text version
        formatted_text = []
        for category, items in shopping_dict.items():
            if items:
                formatted_text.append(f"\n【 {category.upper()} 】")
                for ing_name, details in items.items():
                    qty = details['quantity']
                    unit = details['unit']
                    note = details['note']
                    line = f"  ☐ {qty} {unit} {ing_name}".replace('  ☐ To taste  ', '  ☐ ')
                    if note:
                        line += f" {note}"
                    formatted_text.append(line)
        
        result['formatted_text'] = '\n'.join(formatted_text)
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to prepare shopping list: {str(e)}"})

# --- 3. THE FOOD PLANNER AGENT ---

@dataclass
class FoodPlanner(Agent):
    def __post_init__(self):
        super().__post_init__()
        self.tools.append(
            function_tool(
                google_search_grounded_sync,
                name_override="search_web",
                strict_mode=False,
            )
        )
        self.tools.append(function_tool(get_local_recipe_type, strict_mode=False))
        self.tools.append(function_tool(fetch_local_recipe, strict_mode=False))
        self.tools.append(function_tool(check_cfia_recalls, strict_mode=False))
        self.tools.append(function_tool(modify_recipe, strict_mode=False))
        self.tools.append(function_tool(prepare_shopping_list, strict_mode=False))
