import csv
import pickle
import os
import logging
import numpy as np
from openai import OpenAI
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, )


def read_csv_file(csv_file_path):
    show_list = []
    show_names = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for show in csv_reader:
            show_list.append(show)
            show_names.append(show[0])

    return show_list, show_names


def make_pickle_file(my_dict, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(my_dict, file)


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


def create_embeddings_vectors(show_list):
    show_vectors_dict = {}

    for show in show_list:
        show_name = show[0]
        show_description = show[1]
        response = client.embeddings.create(
            input=show_description,
            model="text-embedding-ada-002"
        )
        show_vectors_dict[show_name] = response.data[0].embedding

    make_pickle_file(show_vectors_dict, './embeddings_dict.pkl')


def get_user_shows():
    while True:
        user_shows_input = input(
            "Which TV shows did you love watching? Separate them by a comma. Make sure to enter more than 1 show: \n")
        user_shows_list = validate_user_shows(user_shows_input)
        shows = ", ".join(user_shows_list)
        user_approval = input(f"Just to make sure, do you mean {shows}?(y/n): \n")
        if user_approval == "y":
            break
        logging.info("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly")

    return user_shows_list


def validate_user_shows(user_shows_input):
    user_shows_list = user_shows_input.split(",")
    for index, show in enumerate(user_shows_list):
        show = show.strip()
        if show in show_names:
            continue
        else:
            user_shows_list[index] = get_similar_show(show)

    return user_shows_list


def get_similar_show(show):
    best_match, _ = process.extractOne(show, show_names)

    return best_match


def calculate_average_vector(current_shows_list, embeddings):
    embeddings_list = []
    num_of_shows = len(current_shows_list)
    for show_name in current_shows_list:
        embeddings_list.append(embeddings[show_name])
    average_vector = [sum(list) / num_of_shows for list in zip(*embeddings_list)]

    return average_vector


def get_recommendations(user_shows_list, embeddings_dict, average_vector):
    logging.info("Great! Generating recommendations...")
    recommendations_list = []
    for show_name in show_names:
        if show_name in user_shows_list:
            continue
        similarity = cosine_similarity(embeddings_dict[show_name], average_vector)
        recommendations_list.append([similarity, show_name])

    recommendations_list.sort(key=lambda x: x[0])
    recommendations_list = (recommendations_list[-5:])[::-1]

    for recommendation in recommendations_list:
        recommendation[0] = round(recommendation[0], 2) * 100

    return recommendations_list


def generate_new_show(shows_list):
    shows = ", ".join(shows_list)
    prompt = f"Create a new show based on this list of shows: {shows}. Write only the name of the show and the description. try to be short and concise."
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": prompt}
        ], )

    response = chat_completion.choices[0].message.content
    show_name_index = response.find("Name:")
    show_description_index = response.find("Description:")
    show_name = response[show_name_index + 6: show_description_index].strip()
    show_description = response[show_description_index + 13:].strip()

    return show_name, show_description


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def log_recommendations(recommendations_list, first_show, first_show_description, second_show, second_show_description):
    logging.info("Here are the tv shows that i think you would love:")
    for recommendation in recommendations_list:
        logging.info(f"{recommendation[1]} ({recommendation[0]}%)")

    logging.info(f"""I have also created just for you two shows which I think you would love.
Show #1 is based on the fact that you loved the input shows that you
gave me. Its name is {first_show} and it is about: \n {first_show_description}.
Show #2 is based on the shows that I recommended for you. Its name is
{second_show} and it is about: \n {second_show_description}.
Here are also the 2 tv show ads. Hope you like them!""")


def generate_show_image(shows_list):
    shows = ", ".join(shows_list)
    prompt = f"Create a cover for a TV show based on the list: {shows}."
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1, )
    image_url = response.data[0].url

    return image_url


def show_image(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

    image = Image.open(file_path)
    image.show()

show_list, show_names = read_csv_file('./imdb_tvshows - imdb_tvshows.csv')

if __name__ == "__main__":
    # create_embeddings_vectors(show_list)
    embeddings_dict = load_pickle_file('./embeddings_dict.pkl')
    user_shows_list = get_user_shows()
    average_vector = calculate_average_vector(user_shows_list, embeddings_dict)
    recommendations_list = get_recommendations(user_shows_list, embeddings_dict, average_vector)
    show1_name, show1_description = generate_new_show(user_shows_list)
    recommendations_names = [recommendation[1] for recommendation in recommendations_list]
    show2_name, show2_description = generate_new_show(recommendations_names)
    log_recommendations(recommendations_list, show1_name, show1_description, show2_name, show2_description)
    show_image(generate_show_image(user_shows_list), './firstImage.jpg')
    show_image(generate_show_image(recommendations_names), './secondImage.jpg')
