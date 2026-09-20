"""Lexical reporting controls, no rational point acceptance or optimization."""
from pathlib import Path
import tempfile
import unittest
from soplex_detached.archive import lexical_sizes


class Controls(unittest.TestCase):
    def inspect(self,text):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            p=Path(tmp)/'point.txt';p.write_text(text);return lexical_sizes(p)

    def test_exact_boundary_and_fraction_components(self):
        maximum=str((1<<4096)-1)
        r=self.inspect('x0 '+maximum+'\nx1 -1/'+maximum+'\nx2 '+maximum+'0/3\n')
        self.assertEqual(r['listed_coordinates'],3)
        self.assertEqual(r['coordinates_exceeding_serialized_4096_bit_cap'],1)
        self.assertEqual(r['first_oversized']['coordinate'],'x2')

    def test_large_text_never_needs_large_integer_parse(self):
        r=self.inspect('x0 '+'9'*15000+'/1\n')
        self.assertEqual(r['maximum_integer_decimal_digits'],15000)
        self.assertEqual(r['coordinates_exceeding_serialized_4096_bit_cap'],1)

    def test_bad_lexical_output_rejected(self):
        with self.assertRaises(ValueError):self.inspect('x0 1e300\n')


if __name__=='__main__':unittest.main()
