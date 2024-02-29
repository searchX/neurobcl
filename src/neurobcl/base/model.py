import json
from abc import ABC, abstractmethod
from typing import Dict, List

def order_invariant(feature: str, filters: dict = {}):
    """Converts the filters to a string and joins with feature to make it order invariant (Helps in searching the same in the hash table)

    :param feature: The feature name
    :param filters: The filters to be applied
    :type feature: str
    :type filters: dict
    :return: A string representation of the feature and filters
    :rtype: str
    """
    list_filter_ops = []
    for key in filters:
        list_filter_ops.append(key + "_=" + filters[key])

    # Alphabetically sort the filter operations
    list_filter_ops.sort()

    # Convert to string and join with feature
    return feature + "#" + "&".join(list_filter_ops)

class NeuroBucketClassifier(ABC):
    """The NeuroBucketClassifier class is used to classify the data into buckets based on the given filters and percentiles
    """
    def __init__(self, quantile_gap: int, max_depth: int, indexer_hash: dict = {}, filter_features: Dict[str, List] = [], bucket_features: List = []):
        self.quantile_gap = quantile_gap
        self.max_depth = max_depth

        # Our datastore to save all the data
        self.indexer_hash = indexer_hash

        if type(filter_features) == dict:
            self.filter_features = list(filter_features.keys())
        elif type(filter_features) == list:
            self.filter_features = filter_features

        if type(bucket_features) == dict:
            self.bucket_features = list(bucket_features.keys())
        elif type(bucket_features) == list:
            self.bucket_features = bucket_features

    def _get_best_percentile_list(self, target_key: str, filters: dict = {}):
        """Get the best percentile list for the given target key and filters, brute force approach to find the best percentile list
        
        :param target_key: The target key for which the percentile list is to be found
        :param filters: The filters to be applied
        :type target_key: str
        :type filters: dict
        :return: The best percentile list for the given target key and filters
        :rtype: List
        """
        # Try to find all combination of filters and if there exist percentile list for the same
        percentile_list = self.indexer_hash.get(order_invariant(target_key, filters), None)
        if percentile_list is not None:
            return percentile_list

        # If not found then try to find the best percentile list by removing the filters one by one
        for key in filters:
            new_filters = filters.copy()
            del new_filters[key]
            percentile_list = self._get_best_percentile_list(target_key, new_filters)
            if percentile_list is not None:
                return percentile_list

        # If nothing is found then return None
        return None

    def get(self, target_key: str, target_key_bucket: int, operator: str = '<', filters: dict = {}, buckets: List[int] = [25, 25, 25, 25], debug: bool = False):
        """Get the value for the given target key and target key bucket based on the given filters and buckets

        :param target_key: The target key for which the value is to be found
        :param target_key_bucket: The target key bucket for which the value is to be found
        :param operator: The operator to be used, defaults to '<'
        :param filters: The filters to be applied, defaults to {}
        :param buckets: The buckets to be used, defaults to [25, 25, 25, 25]
        :param debug: The debug flag, outputs verbose data, defaults to False
        :type target_key: str
        :type target_key_bucket: int
        :type operator: str, optional
        :type filters: dict, optional
        :type buckets: List[int], optional
        :type debug: bool, optional
        :return: The value at given bucket, including the operator side
        :rtype: int

        :raises ValueError: If the feature is not found in bucket features
        :raises ValueError: If the bucket is out of range
        :raises ValueError: If the sum of all buckets is not 100
        :raises ValueError: If no percentile list found for the given filters

        .. code-block:: python

            classifier = trainer.index()
            classifier.get("age", 1, '>', buckets=[25, 25, 25, 25]) # Returns the minimum age that should be present in the first bucket (ie 0-25% of the data)

        .. code-block:: python

            classifier = trainer.index()
            classifier.get("age", 4, '<', buckets=[25, 25, 25, 25], filters={"country": "India"}) # Returns the maximum age that should be present in the fourth bucket (ie 75-100% of the data) for the country "India"
    
        :note: The operator can be either '<' or '>', if '<' then the upper bound of the bucket is returned, if '>' then the lower bound of the bucket is returned
        """
        if target_key not in self.bucket_features:
            raise ValueError("Feature not found in bucket features")
        if target_key_bucket < 1 or target_key_bucket > len(buckets):
            raise ValueError("Bucket out of range")
        if sum(buckets) != 100:
            raise ValueError("Sum of all buckets should be 100")

        # Try to find all combination of filters and if there exist percentile list for the same
        percentile_list = self._get_best_percentile_list(target_key, filters)
        if percentile_list is None:
            raise ValueError("No percentile list found for the given filters")
        
        if debug:
            print("Percentile list found for the given filters", percentile_list)
        
        steps_to_follow = target_key_bucket
        if operator == '>':
            steps_to_follow -= 1

        # Now find the percentile for the given bucket
        percentile = 0
        for i in range(steps_to_follow):
            percentile += buckets[i]

        if debug:
            print("Percentile to follow", percentile)

        # Calculate index required for the given percentile
        index = int(percentile / self.quantile_gap)
        if debug:
            print("Index to follow", index)

        return percentile_list[index]

    def toJson(self):
        """Converts the classifier to a JSON string
        
        :return: The JSON string representation of the classifier
        :rtype: str
        """
        return json.dumps(self.__dict__)

    @classmethod
    def fromJson(cls, json_str):
        """Converts the JSON string to a classifier

        :param json_str: The JSON string to be converted
        :type json_str: str
        :return: The classifier from the JSON string
        :rtype: :class:`NeuroBucketClassifier`
        """
        data = json.loads(json_str)
        return cls(**data)

    def __str__(self):
        return self.toJson()

class NeuroBucketTrainer(ABC):
    """The NeuroBucketTrainer class is used to train the classifier based on the given data and features (template only)"""
    def __init__(self, quantile_gap: int = 10, max_depth: int = 2):
        if 100 % quantile_gap != 0:
            raise ValueError("Quantile gap should be a factor of 100")

        self.quantile_gap = quantile_gap
        self.max_depth = max_depth

        # Our datastore to save all the data
        self.indexer_hash = {}

    @abstractmethod
    def total_items(self, filters: dict = {}):
        """Get the total items for the given filters
        
        :param filters: The filters to be applied
        :type filters: dict
        :return: The total items for the given filters
        :rtype: int
        """
        raise NotImplementedError("NeuroBucketClassifier():: total_items method is not implemented")
    
    @abstractmethod
    def get_at(self, target_feature: str, rank: int, filters: dict = {}):
        """Get the value at the given rank for the given target feature and filters

        :param target_feature: The target feature for which the value is to be found
        :param rank: The rank at which the value is to be found
        :param filters: The filters to be applied
        :type target_feature: str
        :type rank: int
        :type filters: dict
        :return: The value at the given rank for the given target feature and filters
        :rtype: int
        """
        raise NotImplementedError("NeuroBucketClassifier():: get_at method is not implemented")
    
    @abstractmethod
    def get_non_bucket_features(self):
        """Get the non bucket features

        :return: The non bucket features
        :rtype: dict
        """
        raise NotImplementedError("NeuroBucketClassifier():: get_non_bucket_features is not implemented")
    
    @abstractmethod
    def get_bucket_features(self):
        """Get the bucket features

        :return: The bucket features
        :rtype: List
        """
        raise NotImplementedError("NeuroBucketClassifier():: get_bucket_features is not implemented")

    def _add_to_indexer(self, bucket_feat_name: str, current_filters: dict, percentile_list: List):
        """Add the percentile list to the indexer hash

        :param bucket_feat_name: The bucket feature name
        :param current_filters: The current filters
        :param percentile_list: The percentile list
        :type bucket_feat_name: str
        :type current_filters: dict
        :type percentile_list: List
        """

        # Check if percentile list has any good entries, ie non null
        # if not any(percentile_list):
        #     return

        self.indexer_hash[order_invariant(bucket_feat_name, current_filters)] = percentile_list

    def depth_features_index(self, bucket_feat_name: str, depth: int, current_filters = {}):
        """Recursively index the features based on the given depth and current filters

        :param bucket_feat_name: The bucket feature name
        :param depth: The depth
        :param current_filters: The current filters, defaults to {}
        :type bucket_feat_name: str
        :type depth: int
        :type current_filters: dict, optional
        """
        # It will terminate when either it reaches the lowest depth or all the features are exhausted
        if depth <= 0 or len(current_filters) == len(self.get_non_bucket_features()):
            # We reached the lowest low, finally find percentile for this one XD
            percentile_list = [0] * (int(100 / self.quantile_gap) + 1)
            for percentile in range(0, 101, self.quantile_gap):
                rank = min(int(percentile * self.total_items(current_filters) / 100), self.total_items(current_filters) - 1)
                percentile_list[int(percentile / self.quantile_gap)] = self.get_at(bucket_feat_name, rank, current_filters)

            self._add_to_indexer(bucket_feat_name, current_filters, percentile_list)
            return

        non_bucket_feats = self.get_non_bucket_features()
        # Else add a filter and keep recursing, also dont add filter if it is already present
        for key in non_bucket_feats:
            if key in current_filters:
                continue
            for value in non_bucket_feats[key]:
                current_filters[key] = value
                self.depth_features_index(bucket_feat_name, depth - 1, current_filters)
                del current_filters[key]

    def index(self):
        """Index the classifier based on the given data and features

        :return: The classifier model
        :rtype: :class:`NeuroBucketClassifier`

        .. note:: This can take a while to index the data
        """
        # Lets try to find percentile for each feature at each interval
        for feature in self.get_bucket_features():
            for depth in range(self.max_depth):
                self.depth_features_index(feature, depth)

        # Finally return classifier model
        return NeuroBucketClassifier(self.quantile_gap, self.max_depth, self.indexer_hash, self.get_non_bucket_features(), self.get_bucket_features())