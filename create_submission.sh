#!/bin/bash
set -e

# Check output files exist
if [ ! -f "output/matching_results.tsv" ] || [ ! -f "output/candidate_pairs.tsv" ]; then
    echo "Error: Output files not found in output/. Run the pipeline first:"
    echo "  cd code/business_entity_resolution/src"
    echo "  python pipeline.py --train_dir ../../../dataset/train --test_dir ../../../dataset/test --output_dir ../../../output"
    exit 1
fi

# Validate before packaging
echo "Running validation..."
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

if [ $? -ne 0 ]; then
    echo "Validation FAILED. Fix the issues above before creating a submission."
    exit 1
fi

TEAM_NAME="${1:-my_team}"
ZIP_NAME="${TEAM_NAME}_submission.zip"

echo ""
echo "Creating submission zip: ${ZIP_NAME}"

# Clean old zip if exists
rm -f "${ZIP_NAME}"

# Create zip with correct structure
zip -r "${ZIP_NAME}" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/ \
    Documentation_template.md

echo ""
echo "Submission package created: ${ZIP_NAME}"
echo "Contents:"
unzip -l "${ZIP_NAME}" | tail -n +4 | head -n -2
