import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import unittest

from checked_gate.checker import check
from checked_gate.propose import propose

CTX = {'request_id': 'a'*64, 'ordered_pair': [0, 1],
       'margin_lower_proof': 'b'*64, 'margin_negative_upper_proof': 'c'*64}


class Enclosures(unittest.TestCase):
    def test_endpoints_against_independent_decimal_oracle(self):
        # Decimal is a test oracle only, never part of acceptance.
        for q in [F(-16), F(-3), F(-1, 100000000), F(0), F(1, 7), F(3), F(16)]:
            with self.subTest(q=q), localcontext() as context:
                context.prec = 90
                margin = [str(q), str(q)]
                p = propose(CTX, margin)
                result = check(p, expected_context=CTX, expected_margin=margin)
                lo, hi = map(F, result['gate'])
                value = 1/(1+(-Decimal(q.numerator)/Decimal(q.denominator)).exp())
                self.assertLessEqual(Decimal(lo.numerator)/Decimal(lo.denominator), value)
                self.assertGreaterEqual(Decimal(hi.numerator)/Decimal(hi.denominator), value)

    def test_tie_and_crossing(self):
        p = propose(CTX, ['0', '0'])
        self.assertEqual(check(p, expected_context=CTX, expected_margin=['0', '0'])['gate'],
                         ['1/2', '1/2'])
        q = propose(CTX, ['-1', '1'])
        a, b = map(F, check(q, expected_context=CTX, expected_margin=['-1', '1'])['gate'])
        self.assertLess(a, F(1, 2)); self.assertGreater(b, F(1, 2))

    def test_safety_relevant_control(self):
        # A complete two-expert/one-property constant control: s_a=1/5,
        # s_b=-4/5 and margin=2. Coarse [1/2,1] cannot prove it;
        # the checked weight range can. Not an empirical route-flip claim.
        p = propose(CTX, ['2', '2'])
        lo = F(check(p, expected_context=CTX, expected_margin=['2', '2'])['gate'][0])
        self.assertLessEqual(F(1, 2)-F(4, 5), 0)
        self.assertGreater(lo-F(4, 5), 0)

    def test_mutations_fail_closed(self):
        original = propose(CTX, ['1/10', '6/5'])
        mutations = [
            lambda p: p['context'].update(request_id='d'*64),
            lambda p: p['context'].update(ordered_pair=[1, 0]),
            lambda p: p['context'].update(margin_lower_proof='d'*64),
            lambda p: p.update(margin=['0', '6/5']),
            lambda p: p['settings'].update(degree=15),
            lambda p: p['endpoints'][0].update(rounded='1/5'),
            lambda p: p['endpoints'][0].update(series='1'),
            lambda p: p['endpoints'][0].update(tail='0'),
            lambda p: p['endpoints'][1]['exp_steps'].pop(),
            lambda p: p['endpoints'][1]['exp_steps'][-1].__setitem__(0, '99'),
            lambda p: p['endpoints'][0].update(gate=['1', '1']),
            lambda p: p.update(gate=['1', '1']),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(i=i):
                p = copy.deepcopy(original); mutate(p)
                with self.assertRaises(ValueError):
                    check(p, expected_context=CTX, expected_margin=['1/10', '6/5'])

    def test_admission(self):
        for margin in [['17', '17'], ['1', '-1'], ['NaN', '1'], ['0.1', '1'],
                       ['0', '9'*2501], [0, 1]]:
            with self.subTest(margin=str(margin)[:50]), self.assertRaises(ValueError):
                propose(CTX, margin)


if __name__ == '__main__':
    unittest.main()
