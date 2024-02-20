import unittest
from unittest.mock import Mock
from unittest import TestCase
from neurobcl.main import train_from_dictionary

class TestNikePercentile(TestCase):
    @classmethod
    def setUpClass(cls):
        data = [
            {"company": "ADIDAS", "category": "Shoes", "listPrice": 2000},
            {"company": "ADIDAS", "category": "Bracelets", "listPrice": 500},
            {"company": "NIKE", "category": "Shoes", "listPrice": 300},
            {"company": "NIKE", "category": "Clothes", "listPrice": 100},
            {"company": "PUMA", "category": "Clothes", "listPrice": 50},
            {"company": "PUMA", "category": "Shoes", "listPrice": 25},
            {"company": "PUMA", "category": "Shoes", "listPrice": 12},
            {"company": "NIKE", "category": "Shoes", "listPrice": 6},
            {"company": "NIKE", "category": "Shoes", "listPrice": 3},
        ]

        super(TestNikePercentile, cls).setUpClass()
        cls.indexed_bucket = train_from_dictionary(data, ["company", "category"], ["listPrice"])
    
    def test_get_boundary(self):
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 1, '>'), 3)
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 4, '<'), 2000)

    def test_left_skewed_boundary(self):
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 2, '>', buckets=[100, 0]), 2000)
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 2, '<', buckets=[100, 0]), 2000)

    def test_right_skewed_boundary(self):
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 2, '>', buckets=[0, 100]), 3)
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 2, '<', buckets=[0, 100]), 2000)
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 1, '<', buckets=[0, 100]), 3)
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 1, '>', buckets=[0, 100]), 3)

    def test_median(self):
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 2, '<', buckets=[25, 25, 25, 25]), 50)
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 3, '>', buckets=[25, 25, 25, 25]), 50)

    def test_filter_shoes(self):
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 1, '>', filters={"category": "Shoes"}), 3)
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 4, '<', filters={"category": "Shoes"}), 2000)

    def test_filter_category_cloths(self):
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 1, '>', filters={"category": "Clothes"}), 50)
        self.assertEqual(TestNikePercentile.indexed_bucket.get("listPrice", 4, '<', filters={"category": "Clothes"}), 100)

if __name__ == '__main__':
    unittest.main()