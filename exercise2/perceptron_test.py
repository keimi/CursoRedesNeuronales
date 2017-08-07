import unittest
from perceptron import Perceptron


class MyTestCase(unittest.TestCase):
    def test_and(self):
        per= Perceptron(1,1,-1.5)
        self.assertEqual(1, per.process_input(1,1))
        self.assertEqual(0, per.process_input(1,0))
        self.assertEqual(0, per.process_input(0,1))
        self.assertEqual(0, per.process_input(0,0))

    def test_or(self):
        per=Perceptron(1,1,-0.5)
        self.assertEqual(1, per.process_input(1,1))
        self.assertEqual(1, per.process_input(1,0))
        self.assertEqual(1, per.process_input(0,1))
        self.assertEqual(0, per.process_input(0,0))

    def test_nand(self):
        per=Perceptron(-1,-1,0.5)
        self.assertEqual(0, per.process_input(1,1))
        self.assertEqual(0, per.process_input(1,0))
        self.assertEqual(0, per.process_input(0,1))
        self.assertEqual(1, per.process_input(0,0))

if __name__ == '__main__':
    unittest.main()
