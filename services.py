import json

from embeddings import vector_store


def get_recipes(cursor, query, k=10):
    recipes = vector_store.search_similar_recipes(query, top_k=k)
    recipes_ids = [item["id"] for item in recipes]

    placeholders = ",".join("?" * len(recipes_ids))
    query = f"SELECT id, name, ingredients_raw, steps, description FROM recipes WHERE id IN ({placeholders})"
    cursor.execute(query, recipes_ids)

    results = cursor.fetchall()

    recipes_dict = {}
    for row in results:
        recipe = dict(row)
        recipe["ingredients_raw"] = json.loads(recipe["ingredients_raw"])
        recipe["steps"] = json.loads(recipe["steps"])
        recipes_dict[recipe["id"]] = recipe

    final_recipes = []
    for item in recipes:
        recipe_id = item["id"]
        if recipe_id in recipes_dict:
            full_recipe = recipes_dict[recipe_id]
            full_recipe["score"] = item["score"]

            final_recipes.append(full_recipe)

    return final_recipes
