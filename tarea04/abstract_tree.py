import random
import operator
import copy


class Node:
    val_dict = {'+': operator.add,
                '-': operator.sub,
                '*': operator.mul,
                '/': operator.truediv}

    def __init__(self, val, left=None, right=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

    def evaluate(self):
        if self.left is None and self.left is None:
            return self.val

        keys = list(Node.val_dict.keys())

        if self.val not in keys:
            return self.val(self.left.evaluate(), self.right.evaluate())

        return self.val_dict[self.val](self.left.evaluate(), self.right.evaluate())

    def __str__(self, level=0):
        ret = "\t"*level+str(self.val)+"\n"
        if self.left is not None:
            ret += self.left.__str__(level+1)
        if self.right is not None:
            ret += self.right.__str__(level+1)
        return ret

    @classmethod
    def random_val(cls, min_val=0, max_val=10, is_max_level=False):
        keys = list(Node.val_dict.keys())

        if bool(random.randint(0, 1)) and not is_max_level:
            index = random.randint(0, len(keys) - 1)
            return keys[index]

        return random.uniform(min_val, max_val)

    @classmethod
    def random_node(cls, min_range=0, max_range=10, level=-1, parent=None):
        node = cls(cls.random_val(min_range, max_range, level == 0), parent=parent)
        keys = list(cls.val_dict.keys())
        if node.val in keys:
            node.left = cls.random_node(min_range, max_range, level-1, node)
            node.right = cls.random_node(min_range, max_range, level-1, node)

        return node


class AbstractTree:
    def __init__(self, root):
        self.root = root

    def evaluate(self):
        return self.root.evaluate()

    def random_node(self, node=None):
        if node is None:
            return self.random_node(self.root)
        if node.left is None and node.right is None:
            return node
        # if random function return the actual node
        elif random.uniform(0, 1) > 0.7 and node.val.__class__ != max.__class__:
            return node
        elif node.left is None:
            return self.random_node(node.right)
        elif node.right is None:
            return self.random_node(node.left)
        else:
            return self.random_node(node.left) if random.randint(0, 1) == 0 else self.random_node(node.right)

    def __str__(self):
        return str(self.root)

    @classmethod
    def random_tree(cls, min_range=0, max_range=1, max_level=-1, node_class=Node):
        if max_level <= 0:
            root = node_class(max)
            root.left = node_class.random_node(min_range, max_range, parent=root)
            root.right = node_class.random_node(min_range, max_range, parent=root)
        else:
            root = Node(max)
            root.left = node_class.random_node(min_range, max_range, level=max_level-1, parent=root)
            root.right = node_class.random_node(min_range, max_range, level=max_level-1, parent=root)
        return cls(root)

    @classmethod
    def cross_tree(cls, tree1, tree2):

        copy_tree1 = copy.deepcopy(tree1)

        sub_node1 = copy_tree1.random_node()
        sub_node2 = copy.deepcopy(tree2.random_node())

        parent1 = sub_node1.parent

        if parent1 is None:
            return cls(sub_node2)

        if parent1.left == sub_node1:
            parent1.left = sub_node2
            sub_node2.parent = parent1
        else:
            parent1.right = sub_node2
            sub_node2.parent = parent1

        return copy_tree1



# tree1 = AbstractTree.random_tree(0.01, 10.0, 5)
# tree2 = AbstractTree.random_tree(0.01, 10.0, 5)
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
