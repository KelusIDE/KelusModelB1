import unittest
from kelus.code_readiness import CASES, Structure
from kelus.evaluate import code_result

class ReadinessTests(unittest.TestCase):
    def test_evaluator_accepts_correct_new_specifications_and_rejects_wrong_arity(self):
        goldens=[
            'def add_three(a, b, c):\n    return a + b + c',
            'def cube(x):\n    return x * x * x',
            'def final_price(price, tax_rate):\n    return price * (1 + tax_rate)',
            'def triple_score(score):\n    return score * 3',
        ]
        for (_,_,name,trials),golden in zip(CASES,goldens):
            self.assertTrue(code_result(golden,name,trials),name)
        self.assertFalse(code_result('def add_three(a, b):\n    return a + b',CASES[0][2],CASES[0][3]))
    def test_structure_does_not_treat_prose_mentions_as_elements(self):
        parser=Structure();parser.feed('You need html, h1, button, style and script.')
        self.assertFalse(parser.tags)
        parser.feed('<html><style>button{color:red}</style><body><h1>Counter</h1><button>+</button><script>let count=0;</script></body></html>')
        self.assertIn('button',parser.tags);self.assertTrue(parser.styles);self.assertTrue(parser.scripts)

if __name__=='__main__':unittest.main()
