import unittest
from unittest.mock import Mock
from unittest import TestCase
from neurobcl.main import train_from_dictionary

class TestRatingsPrice(TestCase):
    def test_simple(self):
        # List of dictionaries with keys: productName, rating, price, category
        classifier = train_from_dictionary([
            {"productName": "product1", "rating": 1.5, "price": 100, "category": "cat1"},
            {"productName": "product2", "rating": 2.1, "price": 200, "category": "cat1"},
            {"productName": "product3", "rating": 3.3, "price": 300, "category": "cat2"},
            {"productName": "product4", "rating": 4.8, "price": 400, "category": "cat2"},
        ], ["productName", "category"], ["rating", "price"])

        self.assertEqual(classifier.get("rating", 1, '>'), 1.5)

if __name__ == '__main__':
    unittest.main()