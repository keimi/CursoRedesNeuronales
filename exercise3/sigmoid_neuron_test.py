import unittest
from sigmoid_neuron import SigmoidNeuron


class MyTestCase(unittest.TestCase):
    def test_and(self):
        per= SigmoidNeuron(1,1,-1.5)
        self.assertGreater(per.process_input(1,1),0.5 )
        self.assertLess(per.process_input(1,0), 0.5)
        self.assertLess(per.process_input(0,1), 0.5)
        self.assertLess(per.process_input(0,0), 0.5)

    def test_or(self):
        per=SigmoidNeuron(1,1,-0.5)
        self.assertGreater(per.process_input(1,1),0.5)
        self.assertGreater(per.process_input(1,0), 0.5)
        self.assertGreater(per.process_input(0,1), 0.5)
        self.assertLess(per.process_input(0,0), 0.5)

    def test_nand(self):
        per=SigmoidNeuron(-1,-1,0.5)
        self.assertLess(per.process_input(1,1), 0.5)
        self.assertLess(per.process_input(1,0), 0.5)
        self.assertLess(per.process_input(0,1), 0.5)
        self.assertGreater(per.process_input(0,0), 0.5)

if __name__ == '__main__':
    unittest.main()
