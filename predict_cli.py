import os
import sys
import argparse
import json

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from inference import get_predictor

def print_animal_dossier(idx, inst, total_count):
    breed = inst.get("predicted_breed")
    details = inst.get("breed_details", {})
    box = inst.get("box", [])

    header = f"=== ANIMAL #{idx} OF {total_count} ==="
    print("\n" + header)
    if box:
        print(f"[BOUNDING BOX]    [x1: {box[0]}, y1: {box[1]}, x2: {box[2]}, y2: {box[3]}]")
    if breed:
        print(f"[PREDICTED BREED] {breed.get('name', 'N/A')} ({breed.get('category', 'Bovine')} - {breed.get('sub_category', 'Dairy')})")
        print(f"[CONFIDENCE]      {breed.get('confidence_percent', 'N/A')}%")
        print(f"[ORIGIN]          {breed.get('origin', 'N/A')} ({breed.get('native_region', 'N/A')})")
    elif inst.get("is_bovine") and not inst.get("is_known_breed"):
        print("[NOTICE] Breed is outside our 10-breed database.")
    else:
        print("[ALERT] Non-bovine animal detected in this region.")

    if inst.get("top_candidates"):
        print("\n  [TOP CANDIDATES]")
        for c_idx, cand in enumerate(inst["top_candidates"], 1):
            print(f"    {c_idx}. {cand['display_name']}: {cand['confidence_percent']}%")

    if "milk_production" in details:
        mp = details["milk_production"]
        print(f"  [MILK PRODUCTION] Daily Yield: {mp.get('daily_yield_liters', 'N/A')} | Fat %: {mp.get('fat_percentage', 'N/A')}")

    if "market_price" in details:
        mp = details["market_price"]
        print(f"  [MARKET PRICE]    {mp.get('currency_inr', 'N/A')} ({mp.get('currency_usd', 'N/A')})")

    if "possible_diseases" in details and details["possible_diseases"]:
        print(f"  [KEY HEALTH RISK] {details['possible_diseases'][0]['name']}: {details['possible_diseases'][0]['symptoms']}")


def main():
    parser = argparse.ArgumentParser(description="Predict Cattle & Buffalo Breeds with Multi-Animal Instance Support")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to cattle/buffalo image")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Number of top candidate classes to display")
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")

    args = parser.parse_args()

    try:
        predictor = get_predictor()
        result = predictor.predict(args.image, top_k=args.top_k)

        # 1. Non-Bovine Image Alert
        if not result.get("is_bovine", True):
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return

            print("\n" + "="*70)
            print("[ERROR ALERT] non - bovine image detected")
            print("="*70)
            print("The provided image is not a cattle or buffalo.")
            print("No classification probabilities performed.\n")
            return

        # 2. Out-of-dataset Bovine Breed Notice
        if not result.get("is_known_breed", True) or not result.get("success", True):
            if args.json:
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return

            print("\n" + "="*70)
            print("[NOTICE] the given breed does not exists in our data")
            print("="*70)
            print("A cattle or buffalo was detected, but the breed is not present in our dataset.")
            print("No classification probabilities performed.\n")
            return

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        total_detected = result.get("total_detected", 1)
        instances = result.get("instances", [])

        print("\n" + "="*70)
        print(f"[BOVISTA AI DIAGNOSTIC] Detected {total_detected} Cattle/Buffalo Instance(s)")
        print("="*70)

        if instances and len(instances) > 1:
            for idx, inst in enumerate(instances, 1):
                print_animal_dossier(idx, inst, total_detected)
            print("\n" + "="*70 + "\n")
        else:
            # Single instance full dossier
            inst = instances[0] if instances else result
            breed = result.get("predicted_breed")
            details = result.get("breed_details", {})

            print(f"[PREDICTED BREED] {breed['name']} ({breed['category']} - {breed['sub_category']})")
            print(f"[CONFIDENCE]      {breed['confidence_percent']}%")
            print(f"[ORIGIN]          {breed['origin']} ({breed['native_region']})")
            print("="*70)

            print("\n[TOP CANDIDATES]")
            for idx, cand in enumerate(result["top_candidates"], 1):
                print(f"  {idx}. {cand['display_name']}: {cand['confidence_percent']}%")

            if "lifespan" in details:
                ls = details["lifespan"]
                print("\n[LIFESPAN & REPRODUCTION]")
                print(f"  - Average Lifespan: {ls.get('average_lifespan_years', 'N/A')}")
                print(f"  - Productive Milking Years: {ls.get('productive_lactation_years', 'N/A')}")
                print(f"  - Age at First Calving: {ls.get('age_at_first_calving_months', 'N/A')}")
                print(f"  - Calving Interval: {ls.get('calving_interval_months', 'N/A')}")

            if "milk_production" in details and "milk_quality" in details:
                mp = details["milk_production"]
                mq = details["milk_quality"]
                print("\n[MILK PRODUCTION & QUALITY]")
                print(f"  - Daily Yield: {mp.get('daily_yield_liters', 'N/A')}")
                print(f"  - Lactation Yield: {mp.get('lactation_yield_liters', 'N/A')}")
                print(f"  - Fat %: {mp.get('fat_percentage', 'N/A')} | SNF %: {mp.get('snf_percentage', 'N/A')} | Protein: {mp.get('protein_percentage', 'N/A')}")
                print(f"  - Beta-Casein Type: {mq.get('beta_casein_type', 'N/A')}")
                print(f"  - Best Uses: {mq.get('suitability', 'N/A')}")

            if "market_price" in details:
                mp = details["market_price"]
                print("\n[MARKET VALUATION & ECONOMIC ROI]")
                print(f"  - Estimated Price Range: {mp.get('currency_inr', 'N/A')} ({mp.get('currency_usd', 'N/A')})")
                print(f"  - Milking Animal: {mp.get('milking_cow_price_inr', 'N/A')}")
                print(f"  - Pregnant Heifer: {mp.get('pregnant_heifer_price_inr', 'N/A')}")
                print(f"  - Breeding Bull: {mp.get('pedigree_bull_price_inr', 'N/A')}")
                print(f"  - ROI Analysis: {mp.get('economic_roi', 'N/A')}")

            if "possible_diseases" in details:
                print("\n[COMMON DISEASES & HEALTH RISKS]")
                for d in details["possible_diseases"]:
                    print(f"  - {d['name']} [{d['severity']}]: {d['symptoms']}")

            if "cure_and_treatment" in details:
                cat = details["cure_and_treatment"]
                print("\n[CURES & MEDICAL MANAGEMENT]")
                print(f"  - Emergency First-Aid: {cat.get('emergency_first_aid', 'N/A')}")
                print(f"  - Veterinary Treatments: {', '.join(cat.get('veterinary_medicines', []))}")
                print(f"  - Ethnoveterinary Remedies: {cat.get('ethnoveterinary_remedies', 'N/A')}")

            if "vaccination_schedule" in details:
                print("\n[VACCINATION SCHEDULE & PROTOCOLS]")
                for v in details["vaccination_schedule"]:
                    print(f"  - {v['vaccine']} -> {v['timing']} ({v['importance']})")

            if "maintenance_and_housing" in details:
                mh = details["maintenance_and_housing"]
                df = mh.get("daily_feed_requirements", {})
                print("\n[MAINTENANCE & DAILY FEEDING]")
                print(f"  - Green Fodder: {df.get('green_fodder_kg', 'N/A')}")
                print(f"  - Dry Fodder: {df.get('dry_fodder_kg', 'N/A')}")
                print(f"  - Concentrate Feed: {df.get('concentrate_feed_kg', 'N/A')}")
                print(f"  - Water Intake: {df.get('clean_drinking_water_liters', 'N/A')}")
                print(f"  - Shed Design: {mh.get('housing_and_shed_design', 'N/A')}")
                print(f"  - Summer Cooling: {mh.get('summer_heat_management', 'N/A')}")

            print("\n" + "="*70 + "\n")

    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
