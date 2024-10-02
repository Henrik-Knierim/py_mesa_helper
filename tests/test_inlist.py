# tests/test_inlist.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from mesa_helper import Inlist

class TestInlist(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Check for verbose command line arguments
        cls.verbose = '--verbose' in sys.argv or '-v' in sys.argv

        # define the path to this file
        cls.path = os.path.dirname(os.path.abspath(__file__))

    def setUp(self):

        # Create the inlist instance that we want to change and compare
        self.inlist_path = 'tests/inlist'
        self.inlist = Inlist(name = self.inlist_path, mesa_options_path = 'tests/', verbose = self.verbose)

        self.inlist_compare_path = 'tests/inlist_comparison'
        self.inlist_comparison = Inlist(name = self.inlist_compare_path)
    
    def test_set_options(self):    
        # read options from inlist_comparison
        # TODO: Implement a `read_all` method that reads all options from the inlist_comparison
        options_to_read = [
            'log_directory',
            'mixing_length_alpha',
            'lgT_lo_for_set_new_abundances',
            'use_Ledoux_criterion',
            'num_cells_for_smooth_gradL_composition_term',
            'xa_mesh_delta_coeff(3)',
            'x_ctrl(1)',
            'x_integer_ctrl(2)',
            'x_logical_ctrl(3)',
            'x_character_ctrl(4)',
            ]
        
        options_to_set = {}
        for option in options_to_read:
            options_to_set[option] = self.inlist_comparison.read_option(option)

        # Set the options in the inlist instance
        self.inlist.set_multiple_options(**options_to_set)

        # Check if the files set the same options
        options_that_have_been_set = {}
        for option in options_to_read:
            options_that_have_been_set[option] = self.inlist.read_option(option)
        
        if self.verbose:
            self.print_inlist_files()

        self.inlist.restore_inlist()

        self.assertEqual(options_to_set, options_that_have_been_set)

    def print_inlist_files(self):
        """Print the content of the inlist files."""
        with open(self.inlist_path, 'r') as file:
            inlist_content = file.read()
        with open(self.inlist_compare_path, 'r') as file:
            inlist_compare_content = file.read()
        print(inlist_content)
        print('\n\n')
        print(inlist_compare_content)

    def test_create_logs_path(self):
        """Tests whether the logs path is created correctly."""        

        logs_path = Inlist.create_logs_path(logs_style = None, logs_dir = 'test', verbose = self.verbose)
        expected_path = 'LOGS/test'
        self.assertEqual(logs_path, expected_path)
        print(logs_path) if self.verbose else None

        logs_path = Inlist.create_logs_path(logs_style = 'id', inlist_name = self.inlist_compare_path, option = 'mixing_length_alpha', logs_parent_dir = self.path, verbose = self.verbose)
        expected_path = os.path.join(self.path, '1')
        self.assertEqual(logs_path, expected_path)
        print(logs_path) if self.verbose else None
        # remove folder.index
        os.remove(os.path.join(self.path, 'folder.index'))

        logs_path = Inlist.create_logs_path(logs_style = 's0', s0 = 10, verbose = self.verbose)
        expected_path = 'LOGS/s0_10'
        self.assertEqual(logs_path, expected_path)
        print(logs_path) if self.verbose else None

        logs_path = Inlist.create_logs_path(logs_style=['s0', 'Z'], s0=10, Z=0.02, verbose = self.verbose)
        expected_path = 'LOGS/s0_10_Z_0.02'
        self.assertEqual(logs_path, expected_path)
        print(logs_path) if self.verbose else None

        logs_path = Inlist.create_logs_path(logs_style=['s0', 'Z'], s0=10, Z=0.02, series_dir = 'test', verbose = self.verbose)
        expected_path = 'LOGS/test/s0_10_Z_0.02'
        self.assertEqual(logs_path, expected_path)
        print(logs_path) if self.verbose else None

        logs_path = Inlist.create_logs_path(logs_style=['s0', 'Z'], series_style = 'M_p', M_p = 1.0, s0=10, Z=0.02, verbose = self.verbose)
        expected_path = 'LOGS/M_p_1.00/s0_10_Z_0.02'
        self.assertEqual(logs_path, expected_path)
        print(logs_path) if self.verbose else None


if __name__ == '__main__':
    unittest.main()