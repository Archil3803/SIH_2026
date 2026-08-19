import os
import json
import unittest
import numpy as np
from PIL import Image
from pathlib import Path
from inference import get_predictor
from data_loader import scan_dataset

class TestBovineSystem(unittest.TestCase):
    def test_breed_database_completeness(self):
        db_path = "data/breed_database.json"
        self.assertTrue(os.path.exists(db_path), "breed_database.json must exist")
        
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_breeds = [
            "ayrshire_cattle", "brown_swiss_cattle", "holstein_friesian_cattle",
            "jersey_cattle", "red_dane_cattle", "chhattisgarhi",
            "jaffarabadi", "marathwada", "surti", "toda"
        ]

        self.assertEqual(len(data), 10, f"Expected 10 breeds in DB, found {len(data)}")

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
        dataset_path = Path("dataset")
        self.assertTrue(dataset_path.exists(), "dataset directory must exist")
        
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
        self.assertEqual(res_solid["error"], "non - bovine image detected")
        self.assertNotIn("predicted_breed", res_solid)
        self.assertNotIn("top_candidates", res_solid)

        print("[PASS] Non-bovine rejection & probability suppression verified successfully.")

    def test_in_dataset_breed_success(self):
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
        print("[PASS] In-dataset breed prediction verified successfully.")

if __name__ == "__main__":
    unittest.main()
