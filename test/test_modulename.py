import unittest

from geneinfo.utils import GeneList

class TestGeneList(unittest.TestCase):

    def test_bitwise_or(self):
        self.assertEqual(
             GeneList(['A', 'B']) | GeneList(['C', 'B']),
             GeneList(['A', 'B', 'C']) 
            )
