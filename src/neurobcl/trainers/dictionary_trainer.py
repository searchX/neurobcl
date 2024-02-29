import json
from functools import cache

from neurobcl.base.model import NeuroBucketTrainer
from neurobcl.base.model import NeuroBucketClassifier

def _filter_invariant(item, filter_value):
    if type(item) == list:
        return filter_value in item
    else:
        return item == filter_value
class DictionaryBucketTrainer(NeuroBucketTrainer):
    """Train using dictionary/json data, the format of the data should be a list of dictionaries where each dictionary
    represents a single item and the keys of the dictionary represent the features of the item. (Should be uniform)
    """
    def __init__(self, dict_data, keyword_feats_name, bucket_feats_name, quantile_gap=10, max_depth=2):
        NeuroBucketTrainer.__init__(self, quantile_gap, max_depth)

        self.data = dict_data
        # Iterate over the data and create a list of unique values for each feature
        self.keyword_feats = keyword_feats_name
        self.bucket_feats = bucket_feats_name
        self.keywords = {}
        self.buckets = {}

        def set_add_item(s, item):
            if type(item) == list: # List add
                for i in item:
                    s.add(i)
            else:
                s.add(item)

        for item in self.data:
            for key in self.keyword_feats:
                if key not in self.keywords:
                    self.keywords[key] = set()
                set_add_item(self.keywords[key], item[key])
            for key in self.bucket_feats:
                if key not in self.buckets:
                    self.buckets[key] = set()
                set_add_item(self.buckets[key], item[key])

        # Convert to list (making it easy serializable)
        for key in self.keywords:
            self.keywords[key] = list(self.keywords[key])

        for key in self.buckets:
            self.buckets[key] = list(self.buckets[key])

    @cache
    def _get_filtered_data(self, filters: str = "{}"):
        """Filter the data based on the given filters, cache the result to avoid recomputation."""
        filters = json.loads(filters)

        filtered_data = self.data
        for key in filters:
            filtered_data = [item for item in filtered_data if _filter_invariant(item[key], filters[key])]
        return filtered_data

    @cache
    def _sort_data(self, target_feature: str):
        """Sort the data based on the given target_feature, cache the result to avoid recomputation."""
        return sorted(self.data, key=lambda k: k[target_feature])

    def total_items(self, filters = {}):
        filtered_data = self._get_filtered_data(json.dumps(filters, sort_keys=True))
        return len(filtered_data)
    
    def get_at(self, target_feature, rank, filters = {}):
        """Get the value of the target_feature at the given rank, Use cache precomputations at sorting
        
        :param target_feature: The feature to get the value from
        :param rank: The rank of the value to get
        :param filters: The filters to apply to the data
        :type target_feature: str
        :type rank: int
        :type filters: dict
        :return: The value of the target_feature at the given rank
        :rtype: int"""
        # First sort the data
        sorted_data = self._sort_data(target_feature)
    
        filtered_data = sorted_data
        for key in filters:
            filtered_data = [item for item in filtered_data if _filter_invariant(item[key], filters[key])]

        if len(filtered_data) == 0:
            return None

        return filtered_data[rank][target_feature]
    
    def get_non_bucket_features(self):
        """Get the non-bucket features of the data"""
        return self.keywords
    
    def get_bucket_features(self):
        """Get the bucket features of the data"""
        return self.buckets