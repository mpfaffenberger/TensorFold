"""Host-only checks for the gfx1151 replay fixture and its acceptance criteria."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from check_reference import check_loading_info, compare_case, load_cases


class ReferenceTests(unittest.TestCase):
    def test_committed_traces(self) -> None:
        cases = load_cases(Path(__file__).with_name('gfx1151-reference.json'))
        # Fingerprint only the IDs: reformatting the fixture must not change the test.
        actual = [(case['prompt'], hashlib.sha256(json.dumps(case['token_ids']).encode()).hexdigest())
                  for case in cases]
        self.assertEqual(actual, [
            ('What is 2 + 2? Reply with just the number.',
             'bd09af931c8015676b818d1905158fdf8032596f81fb9260d31716192f182afb'),
            ('What is the capital of Japan? Answer in one word.',
             '8ea7734aa093b19193ad390bbbad2724de8e1e33ac38964e0d2a6f67392ace1d'),
        ])

    def test_reject_invalid_fixtures(self) -> None:
        invalid = [None, {}, [], [None], [{}], [{'prompt': 'x'}],
                   [{'prompt': 'x', 'token_ids': [1]}, {'prompt': 'x', 'token_ids': [2]}]]
        invalid.extend([{'prompt': prompt, 'token_ids': [1]}] for prompt in ('', '  ', None, 2))
        invalid.extend([{'prompt': 'x', 'token_ids': ids}] for ids in (None, [], '1', [True], [1.0], ['1'], [-1]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.json'
            for data in invalid:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_cases(path)

    def test_exact_token_match(self) -> None:
        case = {'prompt': 'sample prompt', 'token_ids': [7, 8, 9]}
        compare_case([7, 8, 9], case)
        for ids in ([7, 0, 9], [8, 7, 9], [7, 8], [7, 8, 9, 10], []):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, 'sample prompt'):
                compare_case(ids, case)

    def test_checkpoint_loading_diagnostics(self) -> None:
        for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
            with self.subTest(key=key):
                check_loading_info({key: [], 'metadata': 'informational'})
                with self.assertRaisesRegex(RuntimeError, key):
                    check_loading_info({key: ['problem']})
