import unittest

from kelus.paint import PaintError, code_block, draw_request, fix_spelling, mix, render, unknown_subject


class PaintTests(unittest.TestCase):
    def test_renders_shapes_loops_and_gradients(self):
        image = render("canvas(64, 48, '#000000')\n"
                       "gradient(0, 0, 64, 24, '#ff0000', '#0000ff')\n"
                       "for i in range(4):\n"
                       "    circle(8 + i * 16, 36, 4, mix('#ffffff', '#00ff00', i / 3))\n"
                       "mountains(40, 10, '#202020', seed=1)\n")
        self.assertEqual(image.size, (64, 48))
        self.assertEqual(image.getpixel((1, 0)), (255, 0, 0))
        self.assertEqual(mix('#000000', '#ffffff', 0.5), '#808080ff')

    def test_same_seed_same_image(self):
        program = "canvas(32, 32)\nfor i in range(20):\n    circle(rand(0, 32), rand(0, 32), 1, '#ffffff')\n"
        self.assertEqual(render(program, seed=3).tobytes(), render(program, seed=3).tobytes())
        self.assertNotEqual(render(program, seed=3).tobytes(), render(program, seed=4).tobytes())

    def test_never_runs_arbitrary_python(self):
        attacks = ["import os\ncanvas(32, 32)",
                   "canvas(32, 32)\nopen('x', 'w')",
                   "canvas(32, 32)\nexec('1')",
                   "canvas(32, 32)\nx = ().__class__",
                   "canvas(32, 32)\ncircle.__globals__",
                   "canvas(32, 32)\ndef f():\n    pass",
                   "canvas(32, 32)\nwhile 1:\n    pass",
                   "canvas(32, 32)\nx = lambda: 1"]
        for source in attacks:
            with self.subTest(source=source), self.assertRaises(PaintError):
                render(source)

    def test_runaway_programs_stop(self):
        for source in ["canvas(32, 32)\nx = 2\nfor i in range(100):\n    x = x * x",
                       "canvas(32, 32)\nfor i in range(100000):\n    for j in range(100000):\n        rect(0, 0, 1, 1, '#ffffff')",
                       "canvas(32, 32)\np = [1]\nfor i in range(40):\n    p = p + p",
                       "canvas(100000, 100000)"]:
            with self.subTest(source=source[:40]), self.assertRaises(PaintError):
                render(source)

    def test_character_parts_compose_a_portrait(self):
        image = render("canvas(640, 400, '#ffffff')\nx = 320\ny = 190\nhair_back(x, y, 'pink', 'twintails')\noutfit(x, y, 'uniform', 'navy')\n"
                       "face(x, y)\nanime_eye(x - 30, y + 6, 'blue', -1)\nanime_eye(x + 30, y + 6, 'blue', 1)\nbangs(x, y, 'pink')\ncat_ears(x, y, 'pink')\n")
        self.assertEqual(image.getpixel((320, 196)), (255, 227, 211))
        self.assertNotEqual(image.getpixel((290, 198)), (255, 227, 211))
        with self.assertRaisesRegex(PaintError, r"^Line 2: hair_back style must be"):
            render("canvas(64, 64)\nhair_back(32, 32, 'pink', 'mohawk')")

    def test_errors_name_the_line(self):
        with self.assertRaisesRegex(PaintError, r'^Line 3: Unknown function spray\(\)'):
            render("canvas(32, 32)\nfor i in range(2):\n    spray(1)\n")

    def test_recognizes_drawing_requests(self):
        self.assertEqual(draw_request('/draw a red balloon'), 'a red balloon')
        self.assertEqual(draw_request('Can you draw the ocean at sunset?'), 'Can you draw the ocean at sunset?')
        self.assertEqual(draw_request('make me a picture of a lighthouse'), 'make me a picture of a lighthouse')
        for text in ['/draw', 'how do I draw a circle in python?', 'what is a drawing?', 'the paint is drying']:
            self.assertIsNone(draw_request(text), text)

    def test_unknown_subjects_get_an_honest_reply(self):
        for text in ['a person', 'a human waving', 'two red cars in the city', 'the ocean at sunset', 'an orange kitten']:
            self.assertIsNone(unknown_subject(text), text)
        for text in ['a dragon', 'my mom', 'a unicorn with wings']:
            self.assertRegex(unknown_subject(text), "hasn't learned to draw that yet.*people, cats, dogs", text)

    def test_typos_in_subjects_are_corrected(self):
        known = {'banana', 'elephant', 'house', 'horse', 'apple'}
        self.assertEqual(fix_spelling('draw me a bnana', known), 'draw me a banana')
        self.assertEqual(fix_spelling('draw an elephnat', known), 'draw an elephant')
        self.assertEqual(fix_spelling('draw a hosue', known), 'draw a house')
        self.assertEqual(fix_spelling('draw a hose', known), 'draw a hose')
        self.assertEqual(fix_spelling('draw an apple, please', known), 'draw an apple, please')

    def test_common_words_are_not_corrected(self):
        known = {'chair', 'tree', 'banana', 'strawberries'}
        common = {'hair', 'three', 'pink'}
        self.assertEqual(fix_spelling('draw a girl with pink hair', known, common), 'draw a girl with pink hair')
        self.assertEqual(fix_spelling('draw three strawberries', known, common), 'draw three strawberries')
        self.assertEqual(fix_spelling('draw a bnana', known, common), 'draw a banana')

        self.assertEqual(fix_spelling('draw three bnanas with pink hair'), 'draw three bananas with pink hair')

    def test_reads_the_code_block_of_a_reply(self):
        self.assertEqual(code_block("Here:\n```python\ncanvas(32, 32)\n```\nDone."), "canvas(32, 32)\n")


if __name__ == '__main__':
    unittest.main()
