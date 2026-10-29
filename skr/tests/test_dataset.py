import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image
from dataset import check_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for split, color in [('train', 'red'), ('val', 'blue')]:
            (self.root / 'images' / split).mkdir(parents=True)
            labels = self.root / 'labels' / split
            labels.mkdir(parents=True)
            Image.new('RGB', (32, 32), color).save(self.root / 'images' / split / 'one.png')
            (labels / 'one.txt').write_text('0 0.5 0.5 0.5 0.5\n')
        self.config = self.root / 'data.yaml'
        self.config.write_text(yaml.safe_dump(dict(path='.', train='images/train', val='images/val', names=['thing'])))

    def test_valid_relative_paths(self):
        data, summary = check_dataset(self.config)
        self.assertEqual(data['path'], str(self.root))
        self.assertEqual(summary['train']['objects'], 1)

    def test_missing_label(self):
        (self.root / 'labels/train/one.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing label'):
            check_dataset(self.config)

    def test_invalid_labels(self):
        for row in ['1 0.5 0.5 0.5 0.5', '0 nan 0.5 0.5 0.5', '0 0.1 0.5 0.8 0.5', '0 0.5 0.5 0 0.5', '0 0.5']:
            with self.subTest(row=row):
                (self.root / 'labels/train/one.txt').write_text(row)
                with self.assertRaises(ValueError):
                    check_dataset(self.config)

    def test_duplicate_content(self):
        (self.root / 'images/val/one.png').write_bytes((self.root / 'images/train/one.png').read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicate image content'):
            check_dataset(self.config)

    def test_background_with_positive_sample(self):
        Image.new('RGB', (32, 32), 'green').save(self.root / 'images/train/background.png')
        (self.root / 'labels/train/background.txt').touch()
        _, summary = check_dataset(self.config)
        self.assertEqual(summary['train']['backgrounds'], 1)


if __name__ == '__main__':
    unittest.main()
