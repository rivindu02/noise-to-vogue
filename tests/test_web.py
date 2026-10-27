"""Dependency-free static integration checks; these are not browser QA."""
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class WebIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "web/index.html").read_text()
        cls.js = (ROOT / "web/app.js").read_text()
        parser = Elements()
        parser.feed(cls.html)
        cls.elements = parser.elements
        cls.ids = [attrs["id"] for _, attrs in cls.elements if "id" in attrs]

    def test_unique_ids_and_all_literal_dom_references_exist(self):
        self.assertEqual(len(self.ids), len(set(self.ids)))
        referenced = set(re.findall(r"\$\(['\"]([^'\"]+)['\"]\)", self.js))
        self.assertFalse(referenced - set(self.ids), f"Missing elements: {referenced - set(self.ids)}")

    def test_local_scripts_and_styles_exist(self):
        for tag, attrs in self.elements:
            url = attrs.get("src") if tag == "script" else attrs.get("href") if tag == "link" else ""
            if url and url.startswith("/web/"):
                self.assertTrue((ROOT / url.lstrip("/")).is_file(), url)

    def test_model_images_have_descriptive_alternative_text(self):
        for tag, attrs in self.elements:
            if tag == "img":
                self.assertGreater(len(attrs.get("alt", "")), 10)

    def test_tab_relationships_and_accessible_control_labels(self):
        for _, attrs in self.elements:
            if attrs.get("role") == "tab":
                self.assertIn(attrs["aria-controls"], self.ids)
        labels = {attrs["for"] for tag, attrs in self.elements if tag == "label" and "for" in attrs}
        # The remaining microscope controls are nested inside their labels.
        for target in ("seed", "schedule", "speed"):
            self.assertIn(target, labels)
        self.assertIn('role="tablist"', self.html)

    def test_no_fabricated_measurements_are_embedded_in_viewer(self):
        self.assertIn("/api/results", self.js)
        self.assertIn("No trained checkpoints", self.js)
        self.assertIn("Training loss is not directly comparable", (ROOT / "README.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
