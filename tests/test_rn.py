# tests/test_rn.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from mesa_helper import Rn

class TestRn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Check for verbose command line arguments
        cls.verbose = '--verbose' in sys.argv or '-v' in sys.argv

        # define the path to this file
        cls.path = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):

        # Create the inlist instance that we want to change and compare
        self.rn_path = 'tests/rn'
        self.rn = Rn(name = self.rn_path, verbose = self.verbose)

        self.rn_compare_path = 'tests/rn_comparison'
        self.rn_compare = Rn(name = self.rn_compare_path)
    
    def test_change_mod_file(self):    
        """Tests whether the rn file is changed correctly."""        
        # TODO: Implement a function that reads what mod file is set in the rn file
        compare_mode_file = 'test.mod'

        self.rn.set_mod_name(compare_mode_file)

        # Check if the files are the same
        # Read both files
        with open(self.rn_path, 'r') as file:
            rn_content = file.read()

        with open(self.rn_compare_path, 'r') as file:
            rn_compare_content = file.read()

        # if verbose print the content of the inlist files
        if self.verbose:
            print(rn_content)
            print('\n\n')
            print(rn_compare_content)

        # Check if the files are the same
        self.assertEqual(rn_content, rn_compare_content)

        # Restore the rn file
        self.rn.restore_rn()


if __name__ == '__main__':
    unittest.main()