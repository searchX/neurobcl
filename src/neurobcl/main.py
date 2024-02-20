from typing import List, Dict

from neurobcl.base.model import NeuroBucketClassifier
from neurobcl.trainers.dictionary_trainer import DictionaryBucketTrainer

def train_from_dictionary(data: List[Dict], keyword_feats_name, bucket_feats_name, quantile_gap=10, max_depth=2) -> NeuroBucketClassifier:
    """Train using dictionary/json data, the format of the data should be a list of dictionaries where each dictionary represents 
    a single item and the keys of the dictionary represent the features of the item. (Should be uniform)

    :param data: The data to train on
    :param keyword_feats_name: The name of the keyword features (categorical features)
    :param bucket_feats_name: The name of the bucket features (numerical features)
    :param quantile_gap: The gap between the quantiles
    :param max_depth: The maximum depth of the decision tree
    :type data: List[Dict]
    :type keyword_feats_name: List[str]
    :type bucket_feats_name: List[str]
    :type quantile_gap: int, optional
    :type max_depth: int, optional
    :return: The trained model
    :rtype: :class:`neurobcl.base.model.NeuroBucketClassifier`

    .. code-block:: python

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

        model = train_from_dictionary(data, ["company", "category"], ["listPrice"])
        print(model.get("listPrice", 1, '>')) # 3
        print(model.get("listPrice", 4, '<')) # 2000
    
    .. note:: The keyword_feats_name and bucket_feats_name should be contained in the data for each item
    """
    trainer = DictionaryBucketTrainer(data, keyword_feats_name, bucket_feats_name, quantile_gap, max_depth)
    return trainer.index()