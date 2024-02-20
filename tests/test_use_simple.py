import unittest
from unittest.mock import Mock
from unittest import TestCase
from neurobcl.main import train_from_dictionary

class TestSimpleUse(TestCase):
    def test_simple(self):
        classifier = train_from_dictionary([
            {"color": "red", "size": "small", "price": 100},
            {"color": "blue", "size": "small", "price": 200},
            {"color": "red", "size": "large", "price": 300},
            {"color": "blue", "size": "large", "price": 400},
        ], ["color", "size"], ["price"])

        self.assertEqual(classifier.get("price", 1, '>'), 100)
        self.assertEqual(classifier.get("price", 4, '<'), 400)
        self.assertEqual(classifier.get("price", 1, '>', filters={"color": "blue"}), 200)

if __name__ == '__main__':
    unittest.main()