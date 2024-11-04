import copy
import os
import random
import threading
import uuid

import torch
from bottle import route, run, static_file, template
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from sortedcontainers import SortedList
from tqdm import tqdm

# Configuration Constants
INITIAL_POPULATION_SIZE = 5
MAX_POPULATION_SIZE = 20

MUTATION_RATE = 0.5
MUTATION_STRENGTH = 0.1
INITIAL_MUTATION_RATE = 1.0
INITIAL_MUTATION_STRENGTH = 1.0
CROSSOVER_RATE = 0.8
PARENT_SELECTION_FACTOR = 2
CULL_SELECTION_FACTOR = 2
COMPETITION_SELECTION_FACTOR = 2

IMAGE_GENERATION_STEPS = 10
POSITIVE_PROMPT = "a mountainous scene"
NEGATIVE_PROMPT = "low quality, low resolution"

SORT_INTERVAL = 10
ELO_K_FACTOR = 64
ELO_INITIAL_RATING = 1500

DATA_FOLDER = "evolver_output"

RANDOM_SEED = 42

population_lock = threading.Lock()

def update_elo_rating(rating, opponent_rating, result, k_factor):
    expected_result = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
    return rating + k_factor * (result - expected_result)

def mutate_embeddings(embeddings, mutation_rate, mutation_strength):
    mutated = {}
    for key, embed in embeddings.items():
        mutation_mask = torch.rand_like(embed) < mutation_rate
        mutation = torch.randn_like(embed) * mutation_strength
        mutated[key] = torch.where(mutation_mask, embed + mutation, embed)
    return mutated

def encode_prompt(positive_prompt, negative_prompt):
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds
    }

def generate_initial_population(size, positive_prompt, negative_prompt, pipeline):
    prompt_embeddings = encode_prompt(positive_prompt, negative_prompt)
    population = SortedList(key=lambda x: x["elo_rating"])
    for _ in tqdm(range(size), desc="Generating initial population"):
        embeddings = mutate_embeddings(prompt_embeddings, INITIAL_MUTATION_RATE, INITIAL_MUTATION_STRENGTH)
        image = pipeline(
            prompt_embeds=embeddings["prompt_embeds"],
            negative_prompt_embeds=embeddings["negative_prompt_embeds"],
            pooled_prompt_embeds=embeddings["pooled_prompt_embeds"],
            negative_pooled_prompt_embeds=embeddings["negative_pooled_prompt_embeds"],
            num_inference_steps=IMAGE_GENERATION_STEPS,
        ).images[0]
        image_file = f"{DATA_FOLDER}/{uuid.uuid4()}.png"
        image.save(image_file)
        population.add({
            "id": uuid.uuid4(),
            "image_file": image_file,
            "embeddings": embeddings,
            "elo_rating": ELO_INITIAL_RATING
        })
    return population

def crossover(parent_1, parent_2, crossover_rate):
    if torch.rand(1) < crossover_rate:
        child_embeddings = {}
        for key in parent_1["embeddings"]:
            mask = torch.rand_like(parent_1["embeddings"][key]) < 0.5
            child_embeddings[key] = torch.where(mask, parent_1["embeddings"][key], parent_2["embeddings"][key])
        return child_embeddings
    else:
        return parent_1["embeddings"] if torch.rand(1) < 0.5 else parent_2["embeddings"]

def select_parents(population, parent_selection_factor):
    parent_1 = next((parent for parent in reversed(population) if torch.rand(1) < parent_selection_factor), population[-1])
    parent_2 = next((parent for parent in reversed(population) if torch.rand(1) < parent_selection_factor), population[-1])
    return parent_1, parent_2

def select_cull(population, cull_selection_factor):
    return next((parent for parent in population if torch.rand(1) < cull_selection_factor), population[0])

def create_child(population, parent_selection_factor, crossover_rate, mutation_rate, mutation_strength, elo_initial_rating, pipeline):
    with population_lock:
        parent_1, parent_2 = select_parents(population, parent_selection_factor)
        parent_1_copy = copy.deepcopy(parent_1)
        parent_2_copy = copy.deepcopy(parent_2)

    child_embeddings = crossover(parent_1_copy, parent_2_copy, crossover_rate)
    mutated_embeddings = mutate_embeddings(child_embeddings, mutation_rate, mutation_strength)

    image = pipeline(
        prompt_embeds=mutated_embeddings["prompt_embeds"],
        negative_prompt_embeds=mutated_embeddings["negative_prompt_embeds"],
        pooled_prompt_embeds=mutated_embeddings["pooled_prompt_embeds"],
        negative_pooled_prompt_embeds=mutated_embeddings["negative_pooled_prompt_embeds"],
        num_inference_steps=IMAGE_GENERATION_STEPS
    ).images[0]

    image_file = f"{DATA_FOLDER}/{uuid.uuid4()}.png"
    image.save(image_file)

    return {
        "id": uuid.uuid4(),
        "image_file": image_file,
        "embeddings": mutated_embeddings,
        "elo_rating": (parent_1_copy["elo_rating"] + parent_2_copy["elo_rating"]) / 2
    }

def cull(population, cull_selection_factor):
    cull_candidate = select_cull(population, cull_selection_factor)
    population.remove(cull_candidate)

def cycle_population(population, parent_selection_factor, crossover_rate, mutation_rate, mutation_strength, elo_initial_rating, max_population_size, cull_selection_factor, pipeline):
    with population_lock:
        if len(population) >= max_population_size:
            cull(population, cull_selection_factor)

    child = create_child(population, parent_selection_factor, crossover_rate, mutation_rate, mutation_strength, elo_initial_rating, pipeline)

    with population_lock:
        population.add(child)

    return population

def find_competition_pair(population, competition_selection_factor):
    with population_lock:
        population_copy = list(population)

    if len(population_copy) < 2:
        return None, None

    min_diff = float('inf')
    closest_pair = None

    for i in range(len(population_copy)):
        for j in range(i + 1, len(population_copy)):
            diff = abs(population_copy[i]['elo_rating'] - population_copy[j]['elo_rating'])
            if diff < min_diff:
                min_diff = diff
                closest_pair = (population_copy[i], population_copy[j])

    return closest_pair

@route('/')
def index():
    with population_lock:
        population_copy = list(population)
    image1, image2 = random.sample(population_copy, 2)
    top_five = population_copy[-5:]
    return template('''
        <h1>Choose the better image</h1>
        <a href="/vote/{{image1}}/{{image2}}"><img src="/image/{{image1}}" style="width:400px"></a>
        <a href="/vote/{{image2}}/{{image1}}"><img src="/image/{{image2}}" style="width:400px"></a>
        <h2>Top 5 Images</h2>
        <div style="display: flex; flex-wrap: wrap;">
            % for img in top_five:
                <div style="margin: 10px;">
                    <img src="/image/{{img['id']}}" style="width:200px">
                    <p>ELO: {{'{:.2f}'.format(img['elo_rating'])}}</p>
                </div>
            % end
        </div>
    ''', image1=image1['id'], image2=image2['id'], top_five=top_five)

@route('/image/<image_id>')
def serve_image(image_id):
    population_copy = list(population)
    image = next(img for img in population_copy if str(img['id']) == image_id)
    return static_file(os.path.basename(image['image_file']), root=DATA_FOLDER)

@route('/vote/<winner>/<loser>')
def vote(winner, loser):
    with population_lock:
        try:
            winner_img = next(img for img in population if str(img['id']) == winner)
            loser_img = next(img for img in population if str(img['id']) == loser)

            winner_rating = update_elo_rating(winner_img['elo_rating'], loser_img['elo_rating'], 1, ELO_K_FACTOR)
            loser_rating = update_elo_rating(loser_img['elo_rating'], winner_img['elo_rating'], 0, ELO_K_FACTOR)

            population.remove(winner_img)
            population.remove(loser_img)

            winner_img['elo_rating'] = winner_rating
            loser_img['elo_rating'] = loser_rating

            population.add(winner_img)
            population.add(loser_img)
        except StopIteration:
            pass

    return index()

def evolution_loop():
    global population
    while True:
        population = cycle_population(
            population,
            PARENT_SELECTION_FACTOR,
            CROSSOVER_RATE,
            MUTATION_RATE,
            MUTATION_STRENGTH,
            ELO_INITIAL_RATING,
            MAX_POPULATION_SIZE,
            CULL_SELECTION_FACTOR,
            pipeline
        )

def main():
    global pipeline, population

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    pipeline = StableDiffusionXLPipeline.from_single_file(
        "./stable_diffusion_xl_model.safetensors",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipeline.enable_sequential_cpu_offload()

    population = generate_initial_population(INITIAL_POPULATION_SIZE, POSITIVE_PROMPT, NEGATIVE_PROMPT, pipeline)

    evolution_thread = threading.Thread(target=evolution_loop)
    evolution_thread.start()

    run(host='localhost', port=8080, debug=True)

if __name__ == "__main__":
    main()
