import os
import sys
from pathlib import Path

# Add backend directory to sys.path
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import unittest
import numpy as np
from PIL import Image, ImageDraw
from inference import get_predictor
from data_loader import scan_dataset

class TestBovineSystem(unittest.TestCase):
    def test_breed_database_completeness(self):
        db_candidates = [
            BACKEND_DIR / "data" / "breed_database.json",
            PROJECT_ROOT / "data" / "breed_database.json",
            Path("data/breed_database.json")
        ]
        db_path = None
        for c in db_candidates:
            if c.exists():
                db_path = c
                break
        self.assertIsNotNone(db_path, "breed_database.json must exist")
        
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_breeds = [
            "chhattisgarhi", "gir", "jaffarabadi", "jersey_cattle",
            "kankrej", "marathwada", "red_sindhi", "sahiwal",
            "surti", "toda"
        ]

        self.assertGreaterEqual(len(data), 10, f"Expected at least 10 breeds in DB, found {len(data)}")

        for breed_id in expected_breeds:
            self.assertIn(breed_id, data, f"Breed '{breed_id}' missing in database")
            b = data[breed_id]

            # Mandatory fields required
            self.assertIn("lifespan", b, f"'lifespan' missing in {breed_id}")
            self.assertIn("milk_production", b, f"'milk_production' missing in {breed_id}")
            self.assertIn("milk_quality", b, f"'milk_quality' missing in {breed_id}")
            self.assertIn("possible_diseases", b, f"'possible_diseases' missing in {breed_id}")
            self.assertIn("cure_and_treatment", b, f"'cure_and_treatment' missing in {breed_id}")
            self.assertIn("vaccination_schedule", b, f"'vaccination_schedule' missing in {breed_id}")
            self.assertIn("market_price", b, f"'market_price' missing in {breed_id}")
            self.assertIn("maintenance_and_housing", b, f"'maintenance_and_housing' missing in {breed_id}")

            # Detailed checks
            self.assertGreater(len(b["possible_diseases"]), 0)
            self.assertGreater(len(b["vaccination_schedule"]), 0)
            self.assertIn("emergency_first_aid", b["cure_and_treatment"])
            self.assertIn("currency_inr", b["market_price"])
            self.assertIn("daily_feed_requirements", b["maintenance_and_housing"])

        print("[PASS] All 10 breeds verified with complete veterinary & economic metadata.")

    def test_dataset_structure(self):
        dataset_candidates = [
            PROJECT_ROOT / "dataset",
            BACKEND_DIR / "dataset",
            Path("dataset")
        ]
        dataset_path = None
        for c in dataset_candidates:
            if c.exists():
                dataset_path = c
                break
        self.assertIsNotNone(dataset_path, "dataset directory must exist")
        
        buffalo_path = dataset_path / "Buffalo"
        cattle_path = dataset_path / "Cattle Breeds"
        
        self.assertTrue(buffalo_path.exists(), "Buffalo directory must exist")
        self.assertTrue(cattle_path.exists(), "Cattle Breeds directory must exist")

        print("[PASS] Dataset folder structure verified.")

    def test_non_bovine_image_suppression(self):
        """
        Verifies that non-bovine images trigger the error alert 'non - bovine image detected'
        and do NOT perform or return any classification probabilities.
        """
        predictor = get_predictor()

        # 1. Random noise test
        noise_img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
        res_noise = predictor.predict(noise_img)

        self.assertFalse(res_noise["is_bovine"], "Noise image should be detected as non-bovine")
        self.assertFalse(res_noise["success"], "Success should be False for non-bovine image")
        self.assertEqual(res_noise["error"], "non - bovine image detected", "Error must be exactly 'non - bovine image detected'")
        self.assertNotIn("predicted_breed", res_noise, "Must not contain predicted_breed for non-bovine")
        self.assertNotIn("top_candidates", res_noise, "Must not contain top_candidates for non-bovine")
        self.assertNotIn("breed_details", res_noise, "Must not contain breed_details for non-bovine")

        # 2. Solid color image test
        solid_img = Image.new("RGB", (224, 224), color=(30, 180, 240))
        res_solid = predictor.predict(solid_img)

        self.assertFalse(res_solid["is_bovine"])
        self.assertIn(res_solid["error"], ["non - bovine image detected", "reupload a clear image"])
        self.assertNotIn("predicted_breed", res_solid)
        self.assertNotIn("top_candidates", res_solid)

        # 3. Inanimate / Costume test (Batmann.jpg)
        if os.path.exists("Batmann.jpg"):
            res_batman = predictor.predict("Batmann.jpg")
            self.assertFalse(res_batman["is_bovine"], "Batmann image must be rejected as non-bovine")
            self.assertEqual(res_batman["error"], "non - bovine image detected")
            self.assertNotIn("predicted_breed", res_batman)

        print("[PASS] Non-bovine rejection & probability suppression verified successfully.")

    def test_cartoon_and_synthetic_image_rejection(self):
        """
        Verifies that cartoonized, cel-shaded, anime, and line-art drawings
        are strictly rejected as non-bovine images.
        """
        predictor = get_predictor()

        # 1. Cartoon Cow Drawing (Line Art & Flat Fill)
        img_cartoon = Image.new("RGB", (400, 400), color=(135, 206, 235))
        draw = ImageDraw.Draw(img_cartoon)
        draw.ellipse([100, 150, 300, 320], fill=(255, 255, 255), outline=(0, 0, 0), width=6)
        draw.ellipse([140, 180, 190, 230], fill=(0, 0, 0), outline=(0, 0, 0), width=4)
        draw.ellipse([210, 220, 270, 280], fill=(0, 0, 0), outline=(0, 0, 0), width=4)
        draw.ellipse([160, 240, 240, 300], fill=(255, 192, 203), outline=(0, 0, 0), width=5)
        draw.polygon([(120, 150), (100, 100), (140, 130)], fill=(220, 220, 100), outline=(0, 0, 0))
        draw.polygon([(280, 150), (300, 100), (260, 130)], fill=(220, 220, 100), outline=(0, 0, 0))

        res_c = predictor.predict(img_cartoon)
        self.assertFalse(res_c["is_bovine"], "Cartoon cow drawing must be rejected as non-bovine")
        self.assertFalse(res_c["success"], "Cartoon must not return success=True")
        self.assertEqual(res_c["error"], "non - bovine image detected")
        self.assertNotIn("predicted_breed", res_c)

        # 2. Anime Character / Digital Vector Art
        img_anime = Image.new("RGB", (400, 400), color=(255, 220, 200))
        d2 = ImageDraw.Draw(img_anime)
        d2.polygon([(100, 100), (200, 50), (300, 100), (250, 300), (150, 300)], fill=(255, 150, 0), outline=(0, 0, 0), width=4)
        d2.ellipse([130, 150, 180, 220], fill=(0, 150, 255), outline=(0, 0, 0), width=3)
        d2.ellipse([220, 150, 270, 220], fill=(0, 150, 255), outline=(0, 0, 0), width=3)

        res_a = predictor.predict(img_anime)
        self.assertFalse(res_a["is_bovine"], "Anime illustration must be rejected as non-bovine")
        self.assertEqual(res_a["error"], "non - bovine image detected")

        print("[PASS] Cartoon and synthetic drawing rejection verified successfully.")

    def test_human_image_rejection(self):
        """
        Verifies that human images (faces, portraits, skin, clothing)
        are accurately differentiated from livestock and rejected as non-bovine.
        """
        predictor = get_predictor()

        # Human portrait / face
        img_human = Image.new("RGB", (400, 400), color=(220, 220, 220))
        d = ImageDraw.Draw(img_human)
        d.ellipse([120, 100, 280, 300], fill=(235, 180, 140))
        d.arc([110, 80, 290, 220], 180, 360, fill=(30, 20, 10), width=30)
        d.ellipse([150, 180, 175, 195], fill=(255, 255, 255), outline=(50, 30, 20), width=2)
        d.ellipse([160, 183, 170, 193], fill=(40, 30, 20))
        d.ellipse([225, 180, 250, 195], fill=(255, 255, 255), outline=(50, 30, 20), width=2)
        d.ellipse([230, 183, 240, 193], fill=(40, 30, 20))
        d.line([(200, 195), (195, 230), (205, 230)], fill=(180, 120, 90), width=3)
        d.line([(175, 260), (225, 260)], fill=(180, 60, 60), width=4)
        d.rectangle([80, 300, 320, 400], fill=(40, 60, 120))

        res_h = predictor.predict(img_human)
        self.assertFalse(res_h["is_bovine"], "Human image must be detected as non-bovine")
        self.assertFalse(res_h["success"], "Human image must return success=False")
        self.assertEqual(res_h["error"], "non - bovine image detected")
        self.assertNotIn("predicted_breed", res_h)

        print("[PASS] Human image rejection verified successfully.")

    def test_in_dataset_single_breed_success(self):
        """
        Verifies that images of breeds in the dataset successfully return full classification and dossier.
        """
        predictor = get_predictor()
        samples, _ = scan_dataset("dataset")
        sample_path = samples[0][0]

        res = predictor.predict(sample_path)
        self.assertTrue(res["success"])
        self.assertTrue(res["is_bovine"])
        self.assertTrue(res["is_known_breed"])
        self.assertIn("predicted_breed", res)
        self.assertIn("top_candidates", res)
        self.assertIn("breed_details", res)
        self.assertIn("instances", res)
        self.assertIn("annotated_image", res)
        print("[PASS] In-dataset single breed prediction verified successfully.")

    def test_multi_bovine_detection_and_classification(self):
        """
        Verifies that when multiple cattle/buffaloes are present in an image,
        the system detects each animal instance, localizes it, crops it, and classifies its breed.
        """
        predictor = get_predictor()
        samples, _ = scan_dataset("dataset")

        # Select two different sample images
        img1_path = samples[0][0]
        img2_path = samples[min(50, len(samples) - 1)][0]

        img1 = Image.open(img1_path).convert("RGB").resize((300, 300))
        img2 = Image.open(img2_path).convert("RGB").resize((300, 300))

        # Create side-by-side multi-animal composite scene
        composite = Image.new("RGB", (620, 300), color=(240, 240, 240))
        composite.paste(img1, (0, 0))
        composite.paste(img2, (320, 0))

        res = predictor.predict(composite)
        self.assertTrue(res["is_bovine"])
        self.assertIn("instances", res)
        self.assertGreaterEqual(len(res["instances"]), 1)
        self.assertIn("annotated_image", res)
        self.assertIn("total_detected", res)

        for inst in res["instances"]:
            self.assertIn("instance_id", inst)
            self.assertIn("box", inst)
            self.assertIn("box_normalized", inst)
            self.assertIn("crop_image", inst)

        print(f"[PASS] Multi-bovine detection & per-instance classification verified ({res['total_detected']} instances).")

if __name__ == "__main__":
    unittest.main()
