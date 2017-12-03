import random
import operator
import copy
from abstract_tree import Node
from abstract_tree import AbstractTree


def div(a, b):
    try:
        value = operator.floordiv(a, b)
    except ZeroDivisionError:
        value = int(1E200)

    return value

class ChNode(Node):

    val_dict = {'+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': div}

    number_set=[]

    @classmethod
    def random_val(cls, min_val, max_val, is_max_level=False):
        keys = list(ChNode.val_dict.keys())

        if bool(random.randint(0, 1)) and not is_max_level:
            index = random.randint(0, len(keys) - 1)
            return keys[index]

        return random.choice(cls.number_set)

class ChTree(AbstractTree):

    @classmethod
    def random_tree(cls, min_range=0, max_range=1, max_level=-1, node_class=ChNode):
        if max_level <= 0:
            root = node_class(random.choice(list(node_class.val_dict.keys())))
            root.left = node_class.random_node(min_range, max_range, parent=root)
            root.right = node_class.random_node(min_range, max_range, parent=root)
        else:
            root = node_class(random.choice(list(node_class.val_dict.keys())))
            root.left = node_class.random_node(min_range, max_range, level=max_level-1, parent=root)
            root.right = node_class.random_node(min_range, max_range, level=max_level-1, parent=root)
        return cls(root)


# ChNode.number_set = [10, 1, 25, 9, 3, 6]
# tree1 = ChTree.random_tree(0.01, 10.0, 5, node_class=ChNode)
# tree2 = ChTree.random_tree(0.01, 10.0, 5, node_class=ChNode)
# print('tree1:')
# print(tree1)
# print("----------------------")
# print('tree2:')
# print(tree2)
# print("----------------------")
#
# hybrid = AbstractTree.cross_tree(tree1, tree2)
#
# print('hybrid')
# print(hybrid)
# print("----------------------")
# print('tree evaluate: ', hybrid.evaluate())
