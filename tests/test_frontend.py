import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "Install frontend requirements to test the Streamlit UI")
class FrontendTests(unittest.TestCase):
    def test_valid_pdf_upload_enables_summary_button(self):
        app_path = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()

        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].disabled)

        app.file_uploader[0].upload(
            "judgment.pdf",
            b"%PDF-1.7 test document",
            "application/pdf",
        ).run()

        self.assertFalse(app.exception)
        self.assertFalse(app.button[0].disabled)


if __name__ == "__main__":
    unittest.main()
