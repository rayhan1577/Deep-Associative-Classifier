#!/usr/bin/env python3

import gc
import sys
import time
import random
import logging
import itertools
from collections import deque, defaultdict, Counter
import os
import psutil
import numpy as np
from scipy.special import gammaln, softmax

import config
from rule import Rule
from tree_layer import _TreeLayer

random.seed(config.random_seed)
np.random.seed(config.random_seed)


class SigDirect:
    def __init__(self, get_logs=None):
        #if get_logs == sys.stderr:
            #logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        #elif get_logs == sys.stdout:
            #logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        #else:
            #logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
        self._big_int = np.int32
    
    def _update_new_layer(self, pair):
        pointer_idx, temp = pair

        valid_indexes = np.where(temp.sum(axis=1)>0)[0]

        global_indexes = np.fromiter(map(self._layers[-1].update_node_np, temp[valid_indexes]), dtype=np.int32)

        self._layers[-2]._pointers[pointer_idx, valid_indexes] = global_indexes

    def _update_new_layer_with_base(self, pair):
        pointer_idx, temp = pair
        
        valid_indexes = np.where(temp.sum(axis=1)>0)[0]

        base_value = self._layers[-1].get_counter() - 1
        self._layers[self.new_depth - 2]._pointers_base[pointer_idx] = base_value

        global_indexes = np.fromiter(map(self._layers[-1].update_node_np, temp[valid_indexes]), dtype=np.int32)

        self._layers[-2]._pointers[pointer_idx, valid_indexes] = global_indexes - base_value

    @staticmethod
    def _get_match_len(x,y):
        for i in range(1, len(x)):
            if len(x[i])!=len(x[0]) or y[i]!=y[0]:
                return i
        return len(x)

    @staticmethod
    def _get_batches(x_, y_):
        batch_sizes = config.BATCH_SIZE
        x,y = zip(*sorted(zip(x_,y_), key=lambda p:(p[1], len(p[0]))))
        i = 0

        while i<len(x):
            batch_size = SigDirect._get_match_len(x[i:i+batch_sizes], y[i:i+batch_sizes])
            yield np.array(x[i:i+batch_size]), y[i]
            i += batch_size


    def _batch_traverse(self, X_transaction, y):
        max_depth = self.new_depth

        first_layer_np = np.zeros((max(self._first_layer.keys())+1), dtype=int)

        for i,j in self._first_layer.items():
            first_layer_np[i] = j

        # for id_items, label_index in zip(batch_Xs, batch_ys):
        for id_items, label_index in SigDirect._get_batches(X_transaction, y):
            first_layer_np_batch =  np.repeat(first_layer_np.reshape(1,-1), len(id_items), axis=0)
            
            node_idxes = np.ones((id_items.shape[0]), dtype=int) * -1
            stack = deque([(node_idxes, id_items, 0, 0)])
            num_items = id_items.shape[1]

            while len(stack)>0:
                node_idxes, id_items, start_index, curr_depth = stack.pop()
                num_data = id_items.shape[0]

                new_depth = curr_depth + 1
                new_layer = self._layers[new_depth - 1]

                if curr_depth == 0:
                    curr_pointer_indexes_batch = first_layer_np_batch
                elif self._layers[curr_depth - 1].has_base():
                    base_value = self._layers[curr_depth - 1]._pointers_base[node_idxes].reshape((-1, 1))
                    curr_pointer_indexes_batch = self._layers[curr_depth - 1]._pointers[node_idxes].astype(self._big_int) + base_value
                else:
                    curr_pointer_indexes_batch = self._layers[curr_depth - 1]._pointers[node_idxes]

                if new_depth == max_depth-1:
                    for i in range(start_index, num_items - max_depth + curr_depth + 1):

                        # get the pointer
                        curr_pointer_indexes = curr_pointer_indexes_batch[np.arange(num_data), id_items[np.arange(num_data),i]]
                        valid_indexes = np.where(~new_layer._pointers_dead[curr_pointer_indexes])[0]
                        for valid_index in valid_indexes:
                            curr_pointer_index = curr_pointer_indexes[valid_index]

                            if self._updates[curr_pointer_index] is None:
                                self._updates[curr_pointer_index] = np.zeros((first_layer_np.shape[0], self._labels_size), dtype=self._best_int) 

                            for idx in range(i+1,len(id_items[0])):
                                if (label_index < 6):
                                    self._updates[curr_pointer_index][id_items[valid_index,idx],label_index] += self._one
                
                else:
                    for i in range(start_index, num_items - max_depth + curr_depth + 1):
                        curr_pointer_indexes = curr_pointer_indexes_batch[np.arange(num_data), id_items[np.arange(num_data),i]]
                        valid_indexes = np.where(~new_layer._pointers_dead[curr_pointer_indexes])[0]
                        if len(valid_indexes)>0:
                            stack.append((curr_pointer_indexes[valid_indexes],id_items[valid_indexes], i+1, new_depth))

    def _process_instance_one(self, instance, label_index):
        for feature in instance:
            if feature in self._first_layer:
                self._layers[0].update_node(self._first_layer[feature], label_index)
            else:
                new_index = self._layers[0].add_node(label_index)
                self._first_layer[feature] = new_index


    def _deepen(self, new_depth, X_transaction, y):
        self.new_depth = new_depth
        tt0 = time.time()
        if new_depth == 1:
            for instance, label_index in zip(X_transaction, y):
                self._process_instance_one(instance, label_index)
        else:
            self._updates = [None] * self._layers[-2]._pointers_dead.shape[0]

            self._batch_traverse(X_transaction, y)

            valid_global_indexes = tuple(filter(lambda x: self._updates[x] is not None, range(len(self._updates))))

            new_size = sum(map(lambda x: np.count_nonzero(self._updates[x].sum(axis=1)), valid_global_indexes))

            if new_size>config.BASE_VALUE_THRESHOLD:
                #logging.info("nodes size: {}".format(new_size))
                pass
            if new_size>config.BASE_VALUE_THRESHOLD:
                self._layers[-2].set_has_base()
            self._layers[-2].add_pointers_capacity()

            self._layers[-1].add_initial_capacity(new_size)

            if new_size>config.BASE_VALUE_THRESHOLD:
                temp = tuple(self._update_new_layer_with_base((x, self._updates[x])) for x in valid_global_indexes)
            else:
                temp = tuple(self._update_new_layer((x, self._updates[x])) for x in valid_global_indexes)

            del self._updates, temp

            # garbage collection
            if self._layers[-1]._data_antecedent.shape[0]>100000:
                gc.collect()

            self._layers[-1].add_final_capacity(new_size)
        """
        logging.info("DEPTH: {:5}, NODES: {:8}, TIME: {:8.2f}".format(
                                                        new_depth, 
                                                        self._layers[-1].get_counter(),
                                                        time.time()-tt0, 
                                                        ))
        """
    def _compute_stats(self, depth):

        curr_layer = self._layers[depth]
        d_lgamma = gammaln(self._database_size + 1)

        temp = np.zeros((curr_layer._data_isminimal.shape[0], 2), dtype=np.float64)
        min_ns = np.zeros((curr_layer._data_isminimal.shape[0]), dtype=self._best_int)
        unions = np.zeros((curr_layer._data_isminimal.shape[0]), dtype=np.int32)

        for label_index in range(self._labels_size):
            label_support = self._get_label_support(label_index)
            
            active_ones = curr_layer._data_subsequent[:,label_index]>0
            inactive_ones   = ~active_ones
            default_ones    = curr_layer._data_antecedent > label_support
            active_ones_pss = np.logical_and(active_ones, curr_layer._data_antecedent - label_support<=0)

            curr_layer._data_ispss[active_ones_pss,label_index] =  \
                                     (gammaln(self._database_size - curr_layer._data_antecedent[active_ones_pss] + 1)
                                        + gammaln(label_support + 1)
                                        - d_lgamma
                                        - gammaln(label_support - curr_layer._data_antecedent[active_ones_pss] + 1)) <= config.ALPHA_LOG
            curr_layer._data_ispss[default_ones,label_index] = True

            IS_PSS = curr_layer._data_ispss[:,label_index] == True

            # set is_minimal
            curr_layer._data_isminimal[active_ones,label_index] = curr_layer._data_antecedent[active_ones] == \
                                                                    curr_layer._data_subsequent[active_ones,label_index]            

            # no pss rule!
            if not IS_PSS.any():
                continue

            # min_n
            min_ns[IS_PSS] = np.minimum(curr_layer._data_antecedent[IS_PSS], label_support) \
                                - curr_layer._data_subsequent[IS_PSS,label_index]

            # union
            unions[IS_PSS] = (curr_layer._data_antecedent[IS_PSS]
                                - curr_layer._data_subsequent[IS_PSS,label_index] + label_support)
                                
            # lz
            lz = gammaln(self._database_size  + 1) \
                                - gammaln(label_support  + 1) \
                                - gammaln(self._database_size - label_support + 1)
            # t1
            temp[IS_PSS,0] = gammaln(curr_layer._data_antecedent[IS_PSS] + 1)
            # t2
            temp[IS_PSS,1] = gammaln(self._database_size - curr_layer._data_antecedent[IS_PSS]  + 1)

            max_n = np.max(min_ns[IS_PSS]).astype(int)
            
            curr_layer._data_ss[~IS_PSS,label_index] = np.inf

            for i in range(max_n+1):
                if i == 0:
                    pss_rns_valid = np.where(IS_PSS)[0]
                else:
                    pss_rns_valid = np.where(min_ns-i>=0)[0]

                l1 = temp[pss_rns_valid,0] \
                        - gammaln(curr_layer._data_subsequent[pss_rns_valid,label_index] + i + 1) \
                        - gammaln(curr_layer._data_antecedent[pss_rns_valid] 
                                - curr_layer._data_subsequent[pss_rns_valid,label_index] - i + 1)
                l2 = temp[pss_rns_valid,1] \
                        - gammaln(self._database_size - unions[pss_rns_valid] + i + 1) \
                        - gammaln(- curr_layer._data_antecedent[pss_rns_valid]
                                     + unions[pss_rns_valid] - i + 1)
                curr_layer._data_ss[pss_rns_valid,label_index] += np.exp(l1 + l2 - lz)

        minimal_nodes_idx = np.where(curr_layer._data_isminimal.any(axis=1) == True)[0]
        nonpss_nodes_idx  = np.where(curr_layer._data_ispss.any(axis=1) ==  False)[0]

        remove_idx = np.union1d(minimal_nodes_idx, nonpss_nodes_idx)
        curr_layer._pointers_dead[remove_idx] = True

        del IS_PSS, default_ones, active_ones_pss, active_ones, inactive_ones
        del temp, min_ns, unions

        return 


    def _set_label_support(self, y):
        unique, counts = np.unique(y, return_counts=True)
        self._label_support = defaultdict(int, zip(unique, counts))

    def _get_label_support(self, label_index=None):
        if label_index is None:
            return np.array([self._label_support[label_index]  for label_index in range(self._labels_size)]).transpose()
        return self._label_support[label_index] 

    def _collect_leaves(self, depth):
        # collect all nodes in last layer (together with their items from previous layers)

        stack  = deque()

        curr_depth = 0

        for item,index in self._first_layer.items():
            stack.append((index, [item], curr_depth))

        last_level_nodes = {}
        parent_nodes     = defaultdict(lambda: None)

        while len(stack)>0:
            index, items, curr_depth = stack.pop()

            if curr_depth == depth-1:
                last_level_nodes[index] = np.array(items, dtype=self._best_int)

            # just traverse!
            else:
                if self._layers[curr_depth]._pointers_dead[index]:
                    continue

                temp = self._layers[curr_depth]._pointers[index].nonzero()[0]
                new_end = temp.shape[0] - (depth-curr_depth-2) + 1

                # this branch's height is not enough to get to (at least) the parents level
                if new_end<=0:
                    continue

                valid_range = temp[:new_end]

                if curr_depth == depth-2:
                    parent_nodes[tuple(items)] = self._layers[-2]._data_minss[index]
                
                if self._layers[curr_depth].has_base():
                    base_value = self._layers[curr_depth]._pointers_base[index]

                    stack.extend([(self._layers[curr_depth]._pointers[index, item].astype(self._big_int) + base_value, items + [item], curr_depth + 1) for
                                  item in valid_range])
                else:
                    stack.extend([(self._layers[curr_depth]._pointers[index, item], items + [item], curr_depth + 1) for
                                  item in valid_range])


        del stack
        return last_level_nodes, parent_nodes

    @staticmethod
    def _get_parents_info(items, parent_leaves):

        # generate all parents
        parents = tuple(itertools.combinations(items, len(items) - 1))

        # all parents are pss, and are not minimal
        if any(parent_items not in  parent_leaves for parent_items in parents):
            return None

        # find and return the min_ss among all parents for all labels
        return np.amin(tuple(map(lambda x:parent_leaves[x], parents)), axis=0)



    def _extract_rules(self, depth):
        # dict of nodes (index --> items) in last layer

        leaves, parent_leaves = self._collect_leaves(depth)

        rules = []
        last_layer = self._layers[-1]

        for index, items in leaves.items():
            if depth == 1:
                parents_info = np.ones((self._labels_size))
            else:
                parents_info = SigDirect._get_parents_info(items, parent_leaves)

            # one parent is not pss or is minimal
            if parents_info is None:
                continue

            last_layer._data_minss[index] = np.minimum(parents_info, last_layer._data_ss[index])

            for label_index in np.where(last_layer._data_ss[index] < parents_info)[0]:                

                # it is not SS
                if not (last_layer._data_ss[index, label_index]<=config.ALPHA):
                    continue

                if last_layer._data_ss[index, label_index] == 0.0:
                    last_layer._data_ss[index, label_index] = 2**-1000

                r = Rule(items, label_index, 
                        last_layer._data_subsequent[index, label_index]/last_layer._data_antecedent[index], 
                        last_layer._data_ss[index, label_index], 
                        last_layer._data_subsequent[index, label_index]/self._database_size)
                rules.append(r)

        del leaves, parent_leaves
        return rules

    def _set_best_int(self, X):
        # how many times at most a feature occures.
        max_count = X.sum(axis=0).max()

        if max_count>=2**30:
            raise Exception("Dataset too big")
        elif max_count>=2**14:
            self._best_int = np.int32
        elif max_count>=2**6:
            self._best_int = np.int16
        else:
            self._best_int = np.int8

    def _sort_input_features(self, X):

        sums = X.sum(axis=0)
        temp1 = sorted(enumerate(sums), key=lambda x:(x[1]), reverse=False)
        temp2 = [x[0] for x in temp1]

        self._sorted_mapping  = dict(enumerate(temp2))

        ret = np.zeros_like(X)
        for new_loc, old_loc in self._sorted_mapping.items():
            ret[:, new_loc] = X[:, old_loc]

        return ret

    def _resort_input_features(self, rules):
        """ given a dictionary of final rules (rule-->count), 
        for each rule, re-map its items to the original values, """

        for rule in rules:
            new_items = sorted(list(map(lambda x:  self._sorted_mapping[x], rule.get_items())))
            rule.set_items(new_items)

        return rules

    def fit(self, X, y,prune=None):
        memory=0
        """ Train a SigDirect classifier. The inputs should be similar to sklearn fit method.

        Args:
        X: train instances in the form of a 2-d numpy array
        y: train labels in the form of a 1-d numpy array

        Returns:
        tuple of number of generated rules, and number of pruned rules.
        """

        #if prune is not None:
            #logging.warning('pruning is disabled.')

        tt0 = time.time()

        # input must be integers
        X = X.astype(int)
        y = y.astype(int)

        y_set = np.unique(y)

        # some checls on user data
        if X.shape[0] != y.shape[0]:
            raise Exception("Size of instances and labels do not match")
        if X.shape[0] == 0:
            raise Exception("Tranining data is empty")
        if len(y_set) == 1:
            raise Exception("All instances belong to one class")
        if X.shape[0] >= 2**31:
            raise Exception("Dataset is too big for this classifier")

        self._set_best_int(X)

        # faster addition when we have the object already built
        self._one = self._best_int(1)

        self._X = X
        self._y = tuple(y)

        # used for pruning
        self._X_transaction_original = [row.nonzero()[0] for row in self._X]

        #logging.info('TRAIN: {} {}'.format(self._X.shape, Counter(self._y)))

        self._X = self._sort_input_features(self._X)

        self._X_transaction = tuple(tuple(row.nonzero()[0]) for row in self._X)
        x_set = set(itertools.chain(*self._X_transaction))

        self._database_size = self._X.shape[0]
        self._labels_size   = len(y_set)
        self._set_label_support(y)

        new_depth = 0
        self._layers = []
        self._first_layer = dict()
        generated_rules = []
        self._label_rules_dict = defaultdict(list)

        # deepen the tree with another layer
        while new_depth<len(x_set):
            new_depth+=1
            self._layers.append(_TreeLayer(self._labels_size, self._best_int, self._X.shape[1]))
            self._deepen(new_depth, self._X_transaction, self._y)
            tt1 = time.time()

            if self._layers[-1]._data_antecedent.sum() == 0:
                break

            self._compute_stats(new_depth-1)
            tt2 = time.time()

            #logging.info('STATS: {:8.2f}'.format(tt2-tt1))

            new_rules = self._extract_rules(new_depth)

            if len(new_rules)==0:
                break

            generated_rules.extend(new_rules)

            # delete references to unused variables
            self._layers[-1].release_initial_capacity()
            if new_depth>=2:
                self._layers[-2].release_further_capacity()

            # garbage collection
            if self._layers[-1]._data_minss.shape[0]>100000:
                gc.collect()
            #output = os.popen('wmic process get description, processid').read()
            #print("no of process",len(output))
            """
            print('RULES: {:7}, MEM: {:6}MB, TIME: {:8.2f}'.format(len(generated_rules), 
                                                                        int(psutil.Process(os.getpid()).memory_info().rss/(10**6)),
                                                                        time.time()-tt2, 
                                                                        ))
            #logging.info('#############################################')
            """
            if(memory<int(psutil.Process().memory_info().rss/(10**6))):
                memory=int(psutil.Process().memory_info().rss/(10**6))
            # no pss node here, end of tree creation
            if not self._layers[-1]._data_ispss.any():
                break

        # re-mapping features
        generated_rules = self._resort_input_features(generated_rules)

        # pruning
        if prune is None:
            self._final_rules = self._prune_rules(generated_rules)
        elif prune == False:
            self._final_rules = {x:1 for x in generated_rules}
        else:
            #logging.warning('incorrect pruning is chosen. switching to default pruning')
            self._final_rules = self._prune_rules(generated_rules)

        # making a dictionary for label --> rules
        self._make_label_rules_dict()

        # release the memory
        del self._layers
        gc.collect()
        
        #logging.info('FINAL RULES: {:6}'.format(len(self._final_rules)))
        #logging.info("TOTAL TIME: {:10.2f}".format(time.time()-tt0))

        return len(generated_rules), len(self._final_rules), memory

    def _make_label_rules_dict(self):
        for rule in self._final_rules:
            self._label_rules_dict[rule.get_label()].append(rule)

    def _prune_rules(self, generated_rules):

        rules_dict = defaultdict(int)
        sorted_rules = sorted(generated_rules, key=lambda rule: (rule.get_items()))
        
        all_rules_label_np = np.array([int(rule.get_label()) for rule in sorted_rules], dtype=int)
        all_rules_conf_np = np.array([rule.get_confidence() for rule in sorted_rules],dtype=np.float64)
        
        max_len = max([len(rule.get_items()) for rule in sorted_rules])
        # add a dummy item (multiple times) to all rules so that they have same length
        all_rules_items = np.array([rule.get_items() + [-10] * (max_len-len(rule.get_items())) for rule in sorted_rules], dtype=int)
        
        # find best rule for each training datapoint (transaction)
        for items, label in zip(self._X_transaction_original, self._y):
            transaction_items_set = np.append(items, [-10]).astype(int)

            match_rules_idx = all_rules_label_np == label
            match_items_idx = np.all(np.isin(all_rules_items, transaction_items_set), axis=1)

            # rules that match (items) and have the same label as the training point
            valid_candid_rules_idx = np.where(np.logical_and(match_rules_idx, match_items_idx))[0]
            if valid_candid_rules_idx.shape[0] == 0:
                continue

            mask = np.zeros_like(all_rules_conf_np, dtype=bool)
            mask[valid_candid_rules_idx] = True
            temp = np.argmax(all_rules_conf_np[mask])
            max_rule_idx = np.arange(all_rules_conf_np.shape[0])[mask][temp]
            max_rule = sorted_rules[max_rule_idx]

            rules_dict[max_rule] += 1

        return rules_dict

    def predict(self, X, heuristic=1):
        """ Given a list of instances, predicts their corresponding class
        labels and returns the labels.

        Args:
        X: test instances in the form of a 2-d numpy array
        heuristic: the heuristic used in classification (1, 2, 3)

        Returns:
        a list of labels corresponding to all instances.
        """

        if heuristic not in (1,2,3):
            raise Exception("heuristic value should either be 1, 2, or 3")

        self._hrs = heuristic

        if type(X) == np.ndarray:
            if len(X.shape) != 2:
                raise Exception("2-d numpy array expected")
            predictions = np.apply_along_axis(self._predict_instance, axis=1, arr=X)
        elif type(X) == list:
            predictions = self._predict_instance(X)
        else:
            raise TypeError("Invalid data type detected in predict function")
        
        return np.array(predictions)

    def predict_proba(self, X, heuristic=1):
        """ Given a list of instances, predicts their corresponding class
        labels and returns the labels.

        Args:
        X: test instances in the form of a 2-d numpy array
        heuristic: the heuristic used in classification (1, 2, 3)

        Returns:
        a list of list of scores corresponding to each instance
        """

        if heuristic not in (1,2,3):
            raise Exception("heuristic value should either be 1, 2, or 3")

        self._hrs = heuristic

        if type(X) == np.ndarray:
            prediction_probs = np.apply_along_axis(self._predict_proba_instance, axis=1, arr=X)
        elif type(X) == list:
            prediction_probs = self._predict_proba_instance(X)
        else:
            raise TypeError("Invalid data type detected in predict_proba")

        return prediction_probs

    def _predict_instance(self, instance):

        hrs = self._hrs

        instance = np.where(np.asarray(instance) == 1)[0]

        # Now, for each label, compute the corresponding score.
        all_labels = self._label_rules_dict.keys()
        scores = [(self._get_similarity_to_label(instance, x, hrs), x) for x in all_labels]

        # find best score based on heuristic
        return SigDirect._get_best_match_label(scores, hrs)

    def _predict_proba_instance(self, instance):

        hrs = self._hrs

        # removing features that are not available in this instance.
        instance = np.where(np.asarray(instance) == 1)[0]

        # Now, for each label, compute the corresponding score.
        all_labels = self._label_rules_dict.keys()
        scores = [self._get_similarity_to_label(instance, x, hrs) for x in all_labels]

        # make probability dist
        scores = softmax(scores)

        return scores

    def _get_similarity_to_label(self, instance, label, hrs):

        heuristic_funcs = [SigDirect._hrs_1, SigDirect._hrs_2, SigDirect._hrs_3]

        sum_ = 0.0
        for rule in self._label_rules_dict[label]:
            if SigDirect._rule_matches(instance, rule):
                sum_ += heuristic_funcs[hrs-1](rule) * self._final_rules[rule]

        return sum_

    @staticmethod
    def _rule_matches(instance, rule):
        instance_items_set = set(instance)
        for id_item in rule.get_items():
            if id_item not in instance_items_set:
                return False
        return True

    @staticmethod
    def _get_best_match_label(scores, hrs):

        min_ = min(scores, key=lambda x:(x[0],x[1]))
        max_ = max(scores, key=lambda x:(x[0],x[1]))

        # these heuristics look for minimum score
        if hrs in [1,3]:
            return min_[1]
        else:
            return max_[1]

    def get_applicable_rules(self, instance):
        """
        Returns all the rules that are applicable to this instance

        Args:
        instance: one instance in the form of a 1-d numpy array or a list

        Returns:
        A list of label-rules pair where for each label,
        we get a list of Rule objects that 'label' is its consequent.
        (the list for a label can be empty) 
        """
        # removing features that are not available in this instance.
        instance = np.where(np.asarray(instance) == 1)[0]

        all_applicable_rules = []
        for label in sorted(self._label_rules_dict):
            applicable_rules = []
            for rule in self._label_rules_dict[label]:
                if self._rule_matches(instance, rule):
                    applicable_rules.append(rule)
            all_applicable_rules.append((label, applicable_rules))
        return all_applicable_rules

    @staticmethod
    def _hrs_1(rule):
        x = rule.get_ss()
        if x> 2*-500:
            return np.log(x)
        else:
            return -float('inf')

    @staticmethod
    def _hrs_2(rule):
        return rule.get_confidence()

    @staticmethod
    def _hrs_3(rule):
        x = rule.get_ss()
        if x > 2*-500:
            return float(np.log(x)) * rule.get_confidence()
        else:
            return -float('inf')
