from ShowSuggesterAI import *
import unittest
import os
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO
import numpy as np

class TestShowSuggester(unittest.TestCase):

    def test_read_csv_file(self):
        show_list, show_names = read_csv_file('./imdb_tvshows - imdb_tvshows.csv')
        self.assertIsNotNone(show_list)
        self.assertIsInstance(show_list, list)
        self.assertIsInstance(show_names, list)
        self.assertGreater(len(show_list), 0)
        self.assertGreater(len(show_names), 0)

    def test_pickle_file_configuration(self):
        data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        make_pickle_file(data, './test.pkl')
        loaded_data = load_pickle_file('./test.pkl')
        self.assertEqual(data, loaded_data)
        self.assertIsInstance(loaded_data, dict)
        self.assertIsNotNone(loaded_data)

    def test_get_similar_show(self):
        test_shows = ["gem of thrones", "The wakling deads", "how i met me mother"]
        test_list = []
        for show in test_shows:
            test_list.append(get_similar_show(show))

        self.assertIsNotNone(test_list)
        self.assertIn("Game of Thrones", test_list)
        self.assertIn("The Walking Dead", test_list)
        self.assertIn("How I Met Your Mother", test_list)

    def test_cosine_similarity(self):
        vector1 = [1, 2, 3]
        vector2 = [4, 5, 6]
        similarity = cosine_similarity(vector1, vector2)
        self.assertAlmostEqual(similarity, 0.974631846)

    def test_calculate_average_vector(self):
        user_shows_list = ['Show1', 'Show2']
        embeddings_dict = {'Show1': [0.1, 0.2, 0.3], 'Show2': [0.4, 0.5, 0.6]}
        average_vector = calculate_average_vector(user_shows_list, embeddings_dict)
        expected = [0.25, 0.35, 0.45]
        self.assertIsNotNone(average_vector)
        for i in range(3):
            self.assertAlmostEqual(average_vector[i], expected[i])

    @patch('ShowSuggesterAI.logging')
    def test_log_recommendations(self, mock_logging):
        recommendations_list = [(90, "Show A"), (85, "Show B")]
        show1_name = "Show C"
        show1_description = "Description for Show C"
        show2_name = "Show D"
        show2_description = "Description for Show D"

        log_recommendations(recommendations_list, show1_name, show1_description, show2_name, show2_description)

        expected = [
            "Here are the tv shows that i think you would love:",
            "Show A (90%)",
            "Show B (85%)",
            f"I have also created just for you two shows which I think you would love.\nShow #1 is based on the fact that you loved the input shows that you\ngave me. Its name is {show1_name} and it is about: \n {show1_description}.\nShow #2 is based on the shows that I recommended for you. Its name is\n{show2_name} and it is about: \n {show2_description}.\nHere are also the 2 tv show ads. Hope you like them!"
        ]
        for call_index, expected in zip(mock_logging.info.call_args_list, expected):
            self.assertEqual(call_index[0][0], expected)

    
    @patch('ShowSuggesterAI.get_user_shows')
    @patch('ShowSuggesterAI.cosine_similarity', side_effect=[0.9, 0.8, 0.9])
    @patch('ShowSuggesterAI.show_names', ['Show A', 'Show B', 'Show C'])
    def test_get_recommendations(self, mock_user_shows, _):
        mock_user_shows.return_value(['Show A'])
        embeddings_dict = {'Show A': [1, 2, 3], 'Show B': [4, 5, 6], 'Show C': [7, 8, 9]}
        average_vector = [4, 5, 6]

        result = get_recommendations(['Show A'], embeddings_dict, average_vector)

        expected_sorted_list = [[90.0, 'Show B'], [80.0, 'Show C']]
        self.assertEqual(result, expected_sorted_list)


if __name__ == '__main__':
    unittest.main()