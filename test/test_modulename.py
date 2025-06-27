import unittest

import geneinfo.plot
import geneinfo.information
import geneinfo.utils
import geneinfo.genelist 
from geneinfo.genelist import GeneList

class TestGeneList(unittest.TestCase):

    def test_bitwise_or(self):
        self.assertEqual(
             GeneList(['A', 'B']) | GeneList(['C', 'B']),
             GeneList(['A', 'B', 'C']) 
            )
