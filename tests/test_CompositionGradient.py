# tests/test_rn.py
import sys
import os

from sympy import content
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from mesa_helper import CompositionGradient

class TestCompGrad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Check for verbose command line arguments
        cls.verbose = '--verbose' in sys.argv or '-v' in sys.argv

        # define the path to this file
        cls.path = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):

        # Create the inlist instance that we want to change and compare
        self.comp_grad = CompositionGradient(M_p = 0.75)

        self.comp_grad_file = "tests/comp_gradient.dat"
        self.comp_grad_comparison_file = "tests/comp_gradient_comparison.dat"

    
    def test_CompFileCreation(self):
        """Tests whether the rn file is changed correctly."""        
        
        self.comp_grad.abu_profile = lambda m, **kwargs: CompositionGradient.lin(m, m_1 = 0.1, m_2 = 0.5, f_1 = 1.0, f_2 = 0.0)
        self.comp_grad.create_relax_inital_composition_file(relax_composition_filename = self.comp_grad_file, n_bins = 20)


        # Check if the files are the same
        # Read both files
        with open(self.comp_grad_file, 'r') as file:
            content = file.read()

        with open(self.comp_grad_comparison_file, 'r') as file:
            content_compare = file.read()

        self.assertEqual(content, content_compare)

        # remove the file
        os.remove(self.comp_grad_file)

if __name__ == '__main__':
    unittest.main()